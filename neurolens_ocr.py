"""
Neuro-Lens: OCR Module (Step 1) — PaddleOCR v3 compatible
==========================================================
Tested with: paddleocr==3.4.0  paddlepaddle==3.x  Python 3.11

Install:
    pip install paddlepaddle paddleocr opencv-python numpy

Run from command line:
    python neurolens_ocr.py --image page.jpg
    python neurolens_ocr.py --image page.jpg --no-preprocess
    python neurolens_ocr.py --image page.jpg --show
    python neurolens_ocr.py --camera

The JSON saved in neurolens_output/ feeds directly into Step 2 (NLP).
"""

import os
import sys
import cv2
import numpy as np
import json
import argparse
from datetime import datetime

# ── Silence ALL paddle noise before importing ─────────────────
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_call_stack_level"] = "0"
import logging
logging.disable(logging.CRITICAL)

from paddleocr import PaddleOCR

logging.disable(logging.NOTSET)


# ═════════════════════════════════════════════════════════════
# 1. IMAGE PREPROCESSING
# ═════════════════════════════════════════════════════════════

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Textbook-optimised preprocessing pipeline (per synopsis methodology):
      1. Upscale small images  — PaddleOCR needs ≥32 px text height
      2. Grayscale
      3. Bilateral denoise     — removes noise, keeps text edges
      4. CLAHE contrast boost  — fixes uneven phone-camera lighting
      5. Deskew                — corrects page tilt
      6. Adaptive threshold    — clean binary text
      7. Convert back to BGR   — PaddleOCR expects 3-channel input
    """
    h, w = image.shape[:2]
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    deskewed = _deskew(enhanced)
    binary   = cv2.adaptiveThreshold(
        deskewed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct page tilt using Hough line analysis."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return gray
    angles = []
    for line in lines[:20]:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)
    if not angles:
        return gray
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ═════════════════════════════════════════════════════════════
# 2. PADDLEOCR ENGINE  ← v3 API (no use_gpu / show_log / use_angle_cls)
# ═════════════════════════════════════════════════════════════

class NeuroLensOCR:
    """
    PaddleOCR v3 wrapper.
    Valid __init__ params confirmed from paddleocr._pipelines.ocr source:
      lang, use_textline_orientation, text_rec_score_thresh,
      text_det_thresh, text_det_box_thresh, text_det_limit_side_len, …
    GPU is selected via device="gpu" passed through **kwargs → base pipeline.
    use_gpu / show_log are NOT accepted in v3 — removed.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        print(f"[Neuro-Lens] Loading PaddleOCR  lang={lang}  gpu={use_gpu}")
        print("[Neuro-Lens] First run downloads ~200 MB of models …\n")

        # device string: "gpu" or "cpu"  (replaces the old use_gpu bool)
        device = "gpu" if use_gpu else "cpu"

        self.ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,   # replaces use_angle_cls
            device=device,                   # replaces use_gpu
            enable_mkldnn=False,             # fixes oneDNN / MKL-DNN crash on Windows
        )
        self.lang = lang
        print("\n[Neuro-Lens] PaddleOCR ready.\n")

    # ── Core extraction ──────────────────────────────────────

    def extract(self, image: np.ndarray,
                confidence_threshold: float = 0.3) -> dict:
        """
        Run OCR on a BGR image array.

        PaddleOCR v3 result is an iterable of OCRResult objects.
        Each result exposes dict-like keys:
            rec_texts  — list of recognised strings
            rec_scores — list of confidence floats
            rec_polys  — list of bounding polygons [[x,y], …]
        """
        results = self.ocr.predict(image)   # predict() is the v3 entry point

        blocks = []
        lines  = []

        for res in results:                 # one res per input image
            if res is None:
                continue

            texts  = res.get("rec_texts",  [])
            scores = res.get("rec_scores", [])
            polys  = res.get("rec_polys",  [])

            for text, score, poly in zip(texts, scores, polys):
                text = str(text).strip()
                if not text or float(score) < confidence_threshold:
                    continue
                blocks.append({
                    "text"      : text,
                    "bbox"      : [[int(p[0]), int(p[1])] for p in poly],
                    "confidence": round(float(score), 3),
                })
                lines.append(text)

        raw_text   = " ".join(lines)
        paragraphs = _group_paragraphs(lines)
        avg_conf   = (round(sum(b["confidence"] for b in blocks) / len(blocks), 3)
                      if blocks else 0.0)

        return {
            "raw_text"  : raw_text,
            "paragraphs": paragraphs,
            "blocks"    : blocks,
            "metadata"  : {
                "word_count"    : len(raw_text.split()),
                "avg_confidence": avg_conf,
                "timestamp"     : datetime.now().isoformat(),
            },
        }

    # ── From file ────────────────────────────────────────────

    def extract_from_file(self, path: str,
                          confidence_threshold: float = 0.3,
                          preprocess: bool = True) -> dict:
        """Load image → optionally preprocess → OCR."""
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        if preprocess:
            image = preprocess_image(image)
        result = self.extract(image, confidence_threshold)
        result["source"] = path
        return result

    # ── From camera ──────────────────────────────────────────

    def capture_from_camera(self, camera_index: int = 0,
                            confidence_threshold: float = 0.3) -> dict:
        """Webcam capture → preprocess → OCR  (synopsis camera interface)."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera.")
        print("[Neuro-Lens] Camera open — SPACE to capture, Q to quit.")
        captured = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            disp = frame.copy()
            cv2.putText(disp, "SPACE=Capture  Q=Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.imshow("Neuro-Lens Camera", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                captured = frame.copy()
                print("[Neuro-Lens] Captured. Running OCR …")
                break
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if captured is None:
            return {}
        result = self.extract(preprocess_image(captured), confidence_threshold)
        result["source"] = "camera"
        return result


# ═════════════════════════════════════════════════════════════
# 3. TEXT POST-PROCESSING
# ═════════════════════════════════════════════════════════════

def _group_paragraphs(lines: list, min_words: int = 5) -> list:
    """
    Merge raw OCR lines into paragraph-level chunks for the NLP step.
    Lines with < min_words are treated as headings (own paragraph entry).
    """
    paragraphs, current = [], []
    for line in lines:
        if len(line.split()) < min_words:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            if line.strip():
                paragraphs.append(line.strip())
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return [p for p in paragraphs if p.strip()]


# ═════════════════════════════════════════════════════════════
# 4. OUTPUT HELPERS
# ═════════════════════════════════════════════════════════════

def save_result(result: dict, output_dir: str = "neurolens_output") -> str:
    """Save OCR result JSON. Step 2 (NLP) reads this file."""
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"ocr_result_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[Neuro-Lens] Saved → {path}")
    return path


def print_summary(result: dict):
    meta = result.get("metadata", {})
    sep  = "=" * 60
    print(f"\n{sep}")
    print("  NEURO-LENS — OCR RESULT")
    print(sep)
    print(f"  Source          : {result.get('source', '')}")
    print(f"  Words found     : {meta.get('word_count', 0)}")
    print(f"  Avg confidence  : {meta.get('avg_confidence', 0):.1%}")
    print(f"  Blocks detected : {len(result.get('blocks', []))}")
    print("-" * 60)
    paras = result.get("paragraphs", [])
    if paras:
        print("  PARAGRAPHS (→ NLP Step 2):\n")
        for i, p in enumerate(paras, 1):
            print(f"  [{i}] {p}\n")
    else:
        print("  No text extracted. Try these flags:")
        print("    --no-preprocess       skip binarisation")
        print("    --confidence 0.1      lower threshold")
        print("    --show                inspect preprocessed image")
    print(sep)


# ═════════════════════════════════════════════════════════════
# 5. MANUAL TEXT → JSON  (no OCR needed)
# ═════════════════════════════════════════════════════════════

def manual_text_to_result(raw_text: str) -> dict:
    """
    Convert manually typed / pasted text into the same JSON structure
    that the OCR pipeline produces, so Step 2 (NLP) receives identical input.

    Each blank line in the input is treated as a paragraph separator.
    Single-line entries with < 5 words become heading paragraphs.
    """
    # Split on blank lines to get natural paragraphs
    blocks_raw = [b.strip() for b in raw_text.split("\n\n") if b.strip()]

    # Within each block, collapse internal newlines to spaces
    paragraphs = [" ".join(b.split()) for b in blocks_raw if b.strip()]

    joined = " ".join(paragraphs)

    return {
        "raw_text"  : joined,
        "paragraphs": paragraphs,
        "blocks"    : [],           # no bbox data for manual input
        "metadata"  : {
            "word_count"    : len(joined.split()),
            "avg_confidence": 1.0,  # manual = perfect confidence
            "timestamp"     : datetime.now().isoformat(),
            "source_type"   : "manual",
        },
        "source": "manual_input",
    }


# ═════════════════════════════════════════════════════════════
# 6. GUI  (two tabs: OCR from image  |  Manual text entry)
# ═════════════════════════════════════════════════════════════

def run_gui(ocr: "NeuroLensOCR", confidence: float, preprocess: bool,
            output_dir: str):
    """
    Tabbed Tkinter window:
      Tab 1 — Upload an image and run PaddleOCR
      Tab 2 — Type / paste text yourself (paragraphs separated by blank lines)
    Both tabs produce the same JSON structure for the NLP step.
    """
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, scrolledtext, messagebox
    except ImportError:
        print("[Neuro-Lens] Tkinter not available. Use --image flag instead.")
        return

    # ── Root window ──────────────────────────────────────────
    root = tk.Tk()
    root.title("Neuro-Lens — Step 1: Text Input")
    root.geometry("760x680")
    root.resizable(True, True)

    BG     = "#ffffff"
    ACCENT = "#1a73e8"
    MUTED  = "#555555"
    GREEN  = "#1e7e34"
    FONT   = ("Segoe UI", 10)
    FONT_B = ("Segoe UI", 10, "bold")
    FONT_S = ("Segoe UI", 9)
    MONO   = ("Consolas", 9)

    root.configure(bg=BG)

    # ── Header ───────────────────────────────────────────────
    hdr = tk.Frame(root, bg=ACCENT)
    hdr.pack(fill="x")
    tk.Label(hdr, text="Neuro-Lens  |  Step 1 — Text Extraction",
             bg=ACCENT, fg="white", font=("Segoe UI", 12, "bold"),
             pady=11).pack(side="left", padx=16)

    # ── Notebook (tabs) ──────────────────────────────────────
    style = ttk.Style()
    style.theme_use("default")
    style.configure("TNotebook",        background=BG, borderwidth=0)
    style.configure("TNotebook.Tab",    font=FONT_B, padding=[14, 6],
                    background="#e8eaf6", foreground="#333")
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "white")])

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=0, pady=0)

    # shared status bar at bottom
    status_var = tk.StringVar(value="Choose a tab to get started.")
    status_bar = tk.Label(root, textvariable=status_var, bg="#f1f3f4",
                          fg=MUTED, font=FONT_S, anchor="w", pady=4, padx=12)
    status_bar.pack(fill="x", side="bottom")

    # ─────────────────────────────────────────────────────────
    # TAB 1 — OCR from image
    # ─────────────────────────────────────────────────────────
    tab_ocr = tk.Frame(nb, bg=BG, padx=18, pady=14)
    nb.add(tab_ocr, text="  OCR from Image  ")

    # upload row
    up_row = tk.Frame(tab_ocr, bg=BG)
    up_row.pack(fill="x", pady=(0, 8))

    path_var = tk.StringVar(value="No image selected")
    tk.Label(up_row, textvariable=path_var, bg=BG, fg=MUTED,
             font=FONT_S, anchor="w", wraplength=500).pack(side="left",
                                                            fill="x",
                                                            expand=True)

    def browse():
        ft = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
              ("All files", "*.*")]
        chosen = filedialog.askopenfilename(title="Select textbook image",
                                            filetypes=ft)
        if chosen:
            path_var.set(chosen)
            status_var.set("Image selected — click Run OCR.")
            ocr_out.delete("1.0", "end")

    tk.Button(up_row, text="Upload Image", command=browse,
              bg=ACCENT, fg="white", font=FONT_B,
              relief="flat", padx=14, pady=6,
              cursor="hand2").pack(side="right")

    # options row
    opt_row = tk.Frame(tab_ocr, bg=BG)
    opt_row.pack(fill="x", pady=(0, 8))

    pp_var = tk.BooleanVar(value=preprocess)
    tk.Checkbutton(opt_row, text="Preprocess image",
                   variable=pp_var, bg=BG, font=FONT).pack(side="left")
    tk.Label(opt_row, text="    Min confidence:", bg=BG,
             font=FONT).pack(side="left")
    conf_var = tk.DoubleVar(value=confidence)
    tk.Entry(opt_row, textvariable=conf_var, width=5,
             font=FONT).pack(side="left", padx=4)

    # output
    ocr_out = scrolledtext.ScrolledText(tab_ocr, font=MONO, wrap="word",
                                        height=20, relief="flat",
                                        bg="#f8f9fa", fg="#212529")
    ocr_out.pack(fill="both", expand=True)

    def run_ocr():
        path = path_var.get()
        if path == "No image selected" or not os.path.isfile(path):
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        status_var.set("Running OCR … this may take a moment.")
        root.update()
        ocr_out.delete("1.0", "end")
        try:
            result = ocr.extract_from_file(
                path,
                confidence_threshold=conf_var.get(),
                preprocess=pp_var.get()
            )
            saved = save_result(result, output_dir)
            _show_result(ocr_out, result, saved)
            meta  = result.get("metadata", {})
            status_var.set(
                f"OCR done — {meta.get('word_count', 0)} words, "
                f"{len(result.get('paragraphs', []))} paragraphs. "
                f"Saved to {saved}"
            )
        except Exception as exc:
            ocr_out.insert("end", f"ERROR: {exc}\n")
            status_var.set(f"Error: {exc}")

    ocr_btn_row = tk.Frame(tab_ocr, bg=BG)
    ocr_btn_row.pack(fill="x", pady=(8, 0))
    tk.Button(ocr_btn_row, text="Run OCR", command=run_ocr,
              bg="#0d6efd", fg="white", font=FONT_B,
              relief="flat", padx=20, pady=7,
              cursor="hand2").pack(side="right")
    tk.Button(ocr_btn_row, text="Clear",
              command=lambda: ocr_out.delete("1.0", "end"),
              bg=BG, fg=MUTED, font=FONT, relief="flat",
              padx=12, pady=7, cursor="hand2").pack(side="right", padx=6)

    # ─────────────────────────────────────────────────────────
    # TAB 2 — Manual text input
    # ─────────────────────────────────────────────────────────
    tab_manual = tk.Frame(nb, bg=BG, padx=18, pady=14)
    nb.add(tab_manual, text="  Type / Paste Text  ")

    tk.Label(tab_manual,
             text="Type or paste your text below. Separate paragraphs with a blank line.",
             bg=BG, fg=MUTED, font=FONT_S, anchor="w").pack(fill="x",
                                                              pady=(0, 6))

    # paned: top = input, bottom = output
    paned = tk.PanedWindow(tab_manual, orient="vertical",
                           bg="#dee2e6", sashwidth=6, sashrelief="flat")
    paned.pack(fill="both", expand=True)

    # input pane
    in_frame = tk.Frame(paned, bg=BG)
    paned.add(in_frame, minsize=120)

    tk.Label(in_frame, text="Input text:", bg=BG,
             font=FONT_B, fg="#333", anchor="w").pack(fill="x")
    text_in = scrolledtext.ScrolledText(in_frame, font=("Segoe UI", 10),
                                        wrap="word", height=10,
                                        relief="flat", bg="#fffde7",
                                        fg="#212529", insertbackground="#333",
                                        pady=6, padx=6)
    text_in.pack(fill="both", expand=True, pady=(4, 0))

    # placeholder hint
    HINT = ("Example:\n\n"
            "Photosynthesis is the process by which plants make food using sunlight.\n\n"
            "The process happens mainly in the leaves inside chloroplasts.\n\n"
            "Oxygen is released as a byproduct.")
    text_in.insert("1.0", HINT)
    text_in.config(fg="#aaa")

    def _clear_hint(event):
        if text_in.get("1.0", "end-1c") == HINT:
            text_in.delete("1.0", "end")
            text_in.config(fg="#212529")

    text_in.bind("<FocusIn>", _clear_hint)

    # output pane
    out_frame = tk.Frame(paned, bg=BG)
    paned.add(out_frame, minsize=120)

    tk.Label(out_frame, text="Result:", bg=BG,
             font=FONT_B, fg="#333", anchor="w").pack(fill="x")
    manual_out = scrolledtext.ScrolledText(out_frame, font=MONO,
                                           wrap="word", height=10,
                                           relief="flat", bg="#f8f9fa",
                                           fg="#212529")
    manual_out.pack(fill="both", expand=True, pady=(4, 0))

    def save_manual():
        raw = text_in.get("1.0", "end-1c").strip()
        if not raw or raw == HINT.strip():
            messagebox.showwarning("Empty", "Please type or paste some text first.")
            return
        manual_out.delete("1.0", "end")
        result = manual_text_to_result(raw)
        saved  = save_result(result, output_dir)
        _show_result(manual_out, result, saved)
        status_var.set(
            f"Manual text saved — {result['metadata']['word_count']} words, "
            f"{len(result['paragraphs'])} paragraphs. Saved to {saved}"
        )

    def clear_manual():
        text_in.delete("1.0", "end")
        text_in.config(fg="#212529")
        manual_out.delete("1.0", "end")
        status_var.set("Cleared.")

    man_btn_row = tk.Frame(tab_manual, bg=BG)
    man_btn_row.pack(fill="x", pady=(8, 0))

    tk.Button(man_btn_row, text="Save as JSON  →  NLP Step 2",
              command=save_manual,
              bg=GREEN, fg="white", font=FONT_B,
              relief="flat", padx=20, pady=7,
              cursor="hand2").pack(side="right")

    tk.Button(man_btn_row, text="Clear", command=clear_manual,
              bg=BG, fg=MUTED, font=FONT, relief="flat",
              padx=12, pady=7, cursor="hand2").pack(side="right", padx=6)

    # ─────────────────────────────────────────────────────────
    # Shared helper: render result into a ScrolledText widget
    # ─────────────────────────────────────────────────────────
    def _show_result(widget, result, saved_path):
        meta  = result.get("metadata", {})
        paras = result.get("paragraphs", [])
        src   = result.get("source", "")
        conf  = meta.get("avg_confidence", 1.0)

        lines = [
            f"Source          : {src}",
            f"Words found     : {meta.get('word_count', 0)}",
            f"Avg confidence  : {conf:.1%}",
            f"Paragraphs      : {len(paras)}",
            f"Saved to        : {saved_path}",
            "─" * 58,
            "PARAGRAPHS  (→ NLP Step 2):",
            "",
        ]
        for i, p in enumerate(paras, 1):
            lines.append(f"[{i}]  {p}")
            lines.append("")

        if not paras:
            lines.append("No paragraphs extracted.")
            lines.append("For OCR: try 'No preprocess' or lower confidence.")

        widget.delete("1.0", "end")
        widget.insert("end", "\n".join(lines))

    root.mainloop()


# ═════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Lens Step 1 — OCR or manual text input  (PaddleOCR v3)"
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image",  type=str, help="Path to textbook image (skip GUI)")
    src.add_argument("--camera", action="store_true", help="Capture from webcam")
    src.add_argument("--manual", type=str, metavar="TEXT_FILE",
                     help="Path to a plain .txt file to use as manual input")

    parser.add_argument("--lang",          default="en",
                        help="OCR language  (default: en)")
    parser.add_argument("--confidence",    type=float, default=0.3,
                        help="Min confidence 0–1  (default: 0.3)")
    parser.add_argument("--gpu",           action="store_true",
                        help="Use GPU  (NVIDIA GTX 1650+)")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip image preprocessing")
    parser.add_argument("--show",          action="store_true",
                        help="Show preprocessed image before OCR")
    parser.add_argument("--output-dir",    default="neurolens_output",
                        help="Folder to save JSON result")

    args = parser.parse_args()

    # ── CLI: manual text file ────────────────────────────────
    if args.manual:
        if not os.path.isfile(args.manual):
            print(f"[Neuro-Lens] File not found: {args.manual}")
            return
        with open(args.manual, encoding="utf-8") as f:
            raw = f.read()
        result = manual_text_to_result(raw)
        print_summary(result)
        save_result(result, args.output_dir)
        print("[Neuro-Lens] Done. Load the JSON in Step 2 (NLP).\n")
        return

    # ── CLI: image or camera ─────────────────────────────────
    if args.image or args.camera:
        ocr = NeuroLensOCR(lang=args.lang, use_gpu=args.gpu)

        if args.image:
            if args.show:
                img = cv2.imread(args.image)
                if img is not None:
                    cv2.imshow("Neuro-Lens — Preprocessed",
                               preprocess_image(img))
                    print("[Neuro-Lens] Press any key to continue …")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            result = ocr.extract_from_file(
                args.image,
                confidence_threshold=args.confidence,
                preprocess=not args.no_preprocess
            )
        else:
            result = ocr.capture_from_camera(args.confidence)

        if not result:
            print("[Neuro-Lens] No output produced.")
            return

        print_summary(result)
        save_result(result, args.output_dir)
        print("[Neuro-Lens] Done. Load the JSON in Step 2 (NLP).\n")
        return

    # ── Default: open GUI (both tabs available) ───────────────
    ocr = NeuroLensOCR(lang=args.lang, use_gpu=args.gpu)
    run_gui(ocr,
            confidence=args.confidence,
            preprocess=not args.no_preprocess,
            output_dir=args.output_dir)


if __name__ == "__main__":
    main()
