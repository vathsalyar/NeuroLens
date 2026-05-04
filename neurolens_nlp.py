"""
Neuro-Lens: NLP Module (Step 2)
================================
Reads the OCR JSON → runs the full NLP pipeline → writes simplified JSON

Pipeline (per synopsis):
  [OCR JSON]
      → 1. Sentence segmentation          (spaCy)
      → 2. Complexity analysis            (textstat + spaCy)
      → 3. Sentence splitting             (spaCy dependency parse)
      → 4. Lexical simplification         (NLTK WordNet synonyms, freq-filtered)
      → 5. Bullet restructuring           (key-phrase grouping)
      → 6. Readability validation         (Flesch-Kincaid, Flesch Reading Ease)
      → 7. Extractive summarisation       (Sentence Transformers cosine similarity)
  [Simplified JSON]  →  Step 3 (Visual Generation)

Install:
    pip install spacy nltk textstat sentence-transformers torch
    python -m spacy download en_core_web_sm
    python -m nltk.downloader wordnet averaged_perceptron_tagger punkt

Run:
    python neurolens_nlp.py --input neurolens_output/ocr_result_*.json
    python neurolens_nlp.py --input neurolens_output/ocr_result_*.json --gui
"""

import os
import json
import re
import argparse
from datetime import datetime
from pathlib import Path


from neurolens_genai import NeuroLensGenAI 

# ── lazy imports so GUI opens fast ───────────────────────────
import spacy
import nltk
import textstat
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

# Download required NLTK data silently
for pkg in ["wordnet", "averaged_perceptron_tagger", "punkt",
            "punkt_tab", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# DYSLEXIA-AWARE CONSTANTS
# ─────────────────────────────────────────────────────────────

# Words known to be hard for dyslexic readers (visually similar letters,
# long syllable count, irregular pronunciation).  We replace these first
# before running WordNet substitution.
DYSLEXIA_HARD_WORDS = {
    "approximately"  : "about",
    "consequently"   : "so",
    "subsequently"   : "then",
    "immediately"    : "right away",
    "demonstrate"    : "show",
    "furthermore"    : "also",
    "nevertheless"   : "but",
    "comprehend"     : "understand",
    "comprehension"  : "understanding",
    "utilise"        : "use",
    "utilize"        : "use",
    "sufficient"     : "enough",
    "establish"      : "set up",
    "obtain"         : "get",
    "require"        : "need",
    "however"        : "but",
    "therefore"      : "so",
    "although"       : "even though",
    "through"        : "through",   # keep — mirror-letter word, mark only
    "because"        : "because",
    "significant"    : "important",
    "particular"     : "specific",
    "necessary"      : "needed",
    "implement"      : "use",
    "implementation" : "use",
    "individual"     : "person",
    "indicate"       : "show",
    "proceed"        : "go",
    "possess"        : "have",
    "numerous"       : "many",
    "additional"     : "more",
    "provide"        : "give",
    "consider"       : "think about",
    "referred"       : "called",
    "modification"   : "change",
    "throughout"     : "in all of",
    "environment"    : "place",
    "process"        : "step",
    "structure"      : "shape",
    "function"       : "job",
}

# Flesch Reading Ease target for dyslexic-friendly text
# 70–80 = easy for most adults; 80–90 = ideal for dyslexic readers
FLESCH_TARGET_MIN = 60.0

# Sentences longer than this word count get split
LONG_SENTENCE_THRESHOLD = 18

# Bullets per paragraph (synopsis: "short bullet points")
MAX_BULLETS = 6


# ─────────────────────────────────────────────────────────────
# 1. MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_models():
    """Load spaCy and Sentence Transformer once; reuse everywhere."""
    print("[Neuro-Lens NLP] Loading spaCy en_core_web_sm …")
    nlp = spacy.load("en_core_web_sm")

    print("[Neuro-Lens NLP] Loading Sentence Transformer …")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("[Neuro-Lens NLP] Models ready.\n")
    return nlp, embedder


# ─────────────────────────────────────────────────────────────
# 2. SENTENCE SEGMENTATION
# ─────────────────────────────────────────────────────────────

def segment_sentences(paragraph: str, nlp) -> list[str]:
    """
    Use spaCy sentence boundary detection to split a paragraph
    into individual sentences.  Removes empty strings.
    """
    doc = nlp(paragraph)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


# ─────────────────────────────────────────────────────────────
# 3. COMPLEXITY ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyse_complexity(text: str) -> dict:
    """
    Measure linguistic complexity using textstat.
    Returns a dict of readability scores.

    Metrics chosen for dyslexic relevance (per literature survey):
      - Flesch Reading Ease  (higher = easier; target ≥ 60)
      - Flesch-Kincaid Grade (lower grade = simpler)
      - Dale-Chall score     (uses a list of familiar words)
      - avg sentence length  (target ≤ 18 words for dyslexic readers)
      - syllable count / word (target ≤ 1.5)
    """
    word_count = len(text.split())
    if word_count < 3:
        return {"flesch_ease": 100.0, "fk_grade": 0.0,
                "dale_chall": 0.0, "avg_sentence_len": 0.0,
                "avg_syllables": 1.0, "is_complex": False}

    flesch    = textstat.flesch_reading_ease(text)
    fk_grade  = textstat.flesch_kincaid_grade(text)
    dale      = textstat.dale_chall_readability_score(text)
    avg_sent  = textstat.avg_sentence_length(text)
    avg_syl   = textstat.avg_syllables_per_word(text)
    is_complex = (flesch < FLESCH_TARGET_MIN or
                  avg_sent > LONG_SENTENCE_THRESHOLD or
                  avg_syl > 1.7)

    return {
        "flesch_ease"    : round(flesch, 1),
        "fk_grade"       : round(fk_grade, 1),
        "dale_chall"     : round(dale, 1),
        "avg_sentence_len": round(avg_sent, 1),
        "avg_syllables"  : round(avg_syl, 2),
        "is_complex"     : is_complex,
    }


# ─────────────────────────────────────────────────────────────
# 4. SENTENCE SPLITTING
# ─────────────────────────────────────────────────────────────

def split_long_sentence(sentence: str, nlp) -> list[str]:
    """
    Break a long sentence at conjunctions / subordinating clauses
    using spaCy's dependency parse.  Keeps each part meaningful.

    Rules (dyslexia-aware):
      - Split on coordinating conjunctions (CC): and, but, or, so
      - Split on subordinating conjunctions at clause boundaries
      - Never produce a fragment shorter than 4 words
    """
    words = sentence.split()
    if len(words) <= LONG_SENTENCE_THRESHOLD:
        return [sentence]

    doc = nlp(sentence)
    split_points = []   # token indices where we can split

    for token in doc:
        if token.dep_ == "cc" and token.i > 2:
            split_points.append(token.i)
        elif token.dep_ == "mark" and token.i > 3:
            # subordinating clause marker (because, although, while …)
            split_points.append(token.i)
        elif token.dep_ == "advcl" and token.i > 3:
            split_points.append(token.i)

    if not split_points:
        # Fallback: split at midpoint on a comma
        commas = [i for i, t in enumerate(doc) if t.text == ","]
        mid = len(doc) // 2
        closest = min(commas, key=lambda x: abs(x - mid)) if commas else None
        if closest:
            split_points = [closest + 1]

    if not split_points:
        return [sentence]

    # Build fragments from split points
    fragments = []
    prev = 0
    for sp in sorted(set(split_points)):
        frag = doc[prev:sp].text.strip(" ,")
        if len(frag.split()) >= 4:
            fragments.append(_capitalise(frag) + ".")
        prev = sp

    tail = doc[prev:].text.strip(" ,")
    if len(tail.split()) >= 4:
        fragments.append(_capitalise(tail))

    return fragments if len(fragments) > 1 else [sentence]


def _capitalise(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# ─────────────────────────────────────────────────────────────
# 5. LEXICAL SIMPLIFICATION
# ─────────────────────────────────────────────────────────────

# Common English word list (top-5000 by frequency) — we only substitute
# with synonyms that are IN this set, so replacements are always simpler.
_FREQ_WORDS: set | None = None

def _get_freq_words() -> set:
    """Build a set of common English words from NLTK's Brown corpus word list."""
    global _FREQ_WORDS
    if _FREQ_WORDS is None:
        try:
            from nltk.corpus import brown
            nltk.download("brown", quiet=True)
            from collections import Counter
            counts = Counter(w.lower() for w in brown.words())
            _FREQ_WORDS = {w for w, _ in counts.most_common(5000)}
        except Exception:
            _FREQ_WORDS = set()
    return _FREQ_WORDS


def get_simpler_synonym(word: str, pos_tag: str) -> str | None:
    """
    Return a simpler synonym for a word, or None if no good one exists.

    Strategy (dyslexia-aware):
      1. Check the hard-word dictionary first (hand-curated for dyslexia)
      2. Use WordNet to find synonyms in the same POS
      3. Keep only synonyms that are: single-word, shorter, more frequent
    """
    lower = word.lower()

    # Step 1 — hand-curated dyslexia hard-word map
    if lower in DYSLEXIA_HARD_WORDS:
        replacement = DYSLEXIA_HARD_WORDS[lower]
        if replacement != lower:
            return replacement

    # Step 2 — WordNet synonyms
    wn_pos = _pos_to_wordnet(pos_tag)
    if wn_pos is None:
        return None

    synsets = wordnet.synsets(lower, pos=wn_pos)
    if not synsets:
        return None

    freq_words = _get_freq_words()
    candidates = []

    for synset in synsets[:3]:          # top 3 most common synsets
        for lemma in synset.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            # Only single-word, shorter or equal length, more common
            if (" " not in candidate and
                    candidate != lower and
                    len(candidate) < len(lower) and
                    candidate in freq_words):
                candidates.append(candidate)

    if not candidates:
        return None

    # Pick shortest (= least syllables = easiest for dyslexic readers)
    return min(candidates, key=len)


def _pos_to_wordnet(treebank_tag: str):
    """Map NLTK Treebank POS tag to WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def lexically_simplify(sentence: str) -> tuple[str, list[dict]]:
    """
    Replace complex / hard words with simpler synonyms.
    Returns (simplified_sentence, list_of_substitutions).

    Substitutions are tracked so the NLP output JSON records every
    change made — important for transparency and user trust.
    """
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    result_tokens = list(tokens)
    substitutions = []

    for i, (word, tag) in enumerate(tagged):
        if len(word) < 5 or not word.isalpha():
            continue
        simpler = get_simpler_synonym(word, tag)
        if simpler:
            # Preserve capitalisation
            if word[0].isupper():
                simpler = simpler.capitalize()
            result_tokens[i] = simpler
            substitutions.append({"original": word, "replacement": simpler})

    simplified = _detokenize(result_tokens)
    return simplified, substitutions


def _detokenize(tokens: list) -> str:
    """Rejoin tokens into a readable sentence without extra spaces."""
    text = ""
    for i, tok in enumerate(tokens):
        if tok in {".", ",", "!", "?", ";", ":", "'s", "n't", "'re", "'ve",
                   "'ll", "'d", "'m"} or (i > 0 and tokens[i-1] == "'"):
            text += tok
        elif i == 0:
            text += tok
        else:
            text += " " + tok
    return text.strip()


# ─────────────────────────────────────────────────────────────
# 6. BULLET RESTRUCTURING
# ─────────────────────────────────────────────────────────────

def restructure_to_bullets(sentences: list[str], nlp) -> list[str]:
    """
    Convert simplified sentences into dyslexia-friendly bullet points.

    Dyslexia design principles applied:
      - Each bullet = ONE idea (max ~15 words)
      - Start with an action verb where possible
      - Remove filler phrases ("It is important to note that …")
      - Limit to MAX_BULLETS per paragraph (reduces cognitive load)
    """
    FILLER_PATTERNS = [
        r"^it is (important|worth|interesting) to (note|mention|remember) that\s*",
        r"^in (other words|summary|conclusion|addition|fact),?\s*",
        r"^as (mentioned|stated|discussed),?\s*",
        r"^(basically|essentially|generally|typically),?\s*",
        r"^(this means that|note that|please note),?\s*",
    ]

    bullets = []
    for sent in sentences:
        cleaned = sent.strip()
        for pattern in FILLER_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = _capitalise(cleaned.strip(" .,"))

        # Split overly long bullet further
        words = cleaned.split()
        if len(words) > 20:
            doc = nlp(cleaned)
            for s in doc.sents:
                part = s.text.strip()
                if len(part.split()) >= 4:
                    bullets.append(_capitalise(part))
        else:
            if len(cleaned.split()) >= 3:
                bullets.append(cleaned)

    return bullets[:MAX_BULLETS]


# ─────────────────────────────────────────────────────────────
# 7. READABILITY VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_readability(bullets: list[str]) -> dict:
    """
    Compute final readability scores on the simplified bullet text.
    Flag if the target was not reached so downstream steps can re-try.
    """
    combined = ". ".join(bullets)
    if not combined.strip():
        return {"passed": False, "flesch_ease": 0.0, "fk_grade": 99.0, "note": "No text to evaluate."}

    flesch   = textstat.flesch_reading_ease(combined)
    fk_grade = textstat.flesch_kincaid_grade(combined)
    passed   = flesch >= FLESCH_TARGET_MIN

    return {
        "passed"    : passed,
        "flesch_ease": round(flesch, 1),
        "fk_grade"  : round(fk_grade, 1),
        "note"      : ("Good — dyslexia-friendly readability achieved."
                       if passed else
                       f"Score {flesch:.0f} is below target {FLESCH_TARGET_MIN}. "
                       "Consider further simplification.")
    }


# ─────────────────────────────────────────────────────────────
# 8. EXTRACTIVE SUMMARISATION
# ─────────────────────────────────────────────────────────────

def extractive_summary(sentences: list[str], embedder,
                        top_n: int = 3) -> list[str]:
    """
    Identify the most important sentences using cosine similarity
    to the paragraph centroid (Sentence Transformers).

    Returns top_n sentences ranked by relevance — these become the
    "key idea" bullets shown most prominently to the dyslexic reader.
    """
    if len(sentences) <= top_n:
        return sentences

    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    # Centroid = mean of all sentence embeddings
    centroid   = embeddings.mean(dim=0)
    scores     = util.cos_sim(centroid, embeddings)[0]

    ranked = sorted(range(len(sentences)),
                    key=lambda i: scores[i].item(), reverse=True)
    # Return in original order (not ranked order) to preserve flow
    top_indices = sorted(ranked[:top_n])
    return [sentences[i] for i in top_indices]


# ─────────────────────────────────────────────────────────────
# 9. FULL PARAGRAPH PIPELINE
# ─────────────────────────────────────────────────────────────

def process_paragraph(paragraph: str, nlp, embedder, genai) -> dict:
    """
    Run the complete NLP pipeline on one paragraph.
    Returns a structured dict for the output JSON.
    """
    # ── Step 1: Segment ──────────────────────────────────────
    sentences = segment_sentences(paragraph, nlp)
    if not sentences:
        return {}

    # ── Step 2: Complexity analysis (before simplification) ──
    complexity_before = analyse_complexity(paragraph)

    # ── Step 3 + 4: Split long sentences + lexical simplify ──
    simplified_sentences = []
    all_substitutions    = []

    for sent in sentences:
        fragments = split_long_sentence(sent, nlp)
        for frag in fragments:
            simple, subs = lexically_simplify(frag)
            simplified_sentences.append(simple)
            all_substitutions.extend(subs)

    # ── Step 5: Bullet restructuring ─────────────────────────
    bullets = restructure_to_bullets(simplified_sentences, nlp)

    # ── Step 6: Readability validation ───────────────────────
    readability = validate_readability(bullets)

    # ── Step 7: Extractive summarisation (key ideas) ─────────
    key_ideas = extractive_summary(simplified_sentences, embedder, top_n=3)

    # ── Complexity after simplification ──────────────────────
    complexity_after = analyse_complexity(". ".join(bullets))

    genai_result = genai.process_text(paragraph)
    return {
        "original"             : paragraph,
        "simplified_sentences" : simplified_sentences,
        "bullets"              : bullets,
        "key_ideas"            : key_ideas,
        "genai_output": genai_result,
        "substitutions"        : all_substitutions,
        "complexity_before"    : complexity_before,
        "complexity_after"     : complexity_after,
        "readability_check"    : readability,
    }


# ─────────────────────────────────────────────────────────────
# 10. MAIN PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────

def run_pipeline(ocr_json_path: str, nlp, embedder, genai) -> dict:
    """
    Load OCR JSON → process every paragraph → return full NLP result dict.
    """
    with open(ocr_json_path, encoding="utf-8") as f:
        ocr_data = json.load(f)

    paragraphs = ocr_data.get("paragraphs", [])
    source     = ocr_data.get("source", ocr_json_path)

    print(f"[Neuro-Lens NLP] Processing {len(paragraphs)} paragraph(s) from: {source}")

    processed = []
    for i, para in enumerate(paragraphs, 1):
        if not para.strip():
            continue
        print(f"  Paragraph {i}/{len(paragraphs)} …")
        result = process_paragraph(para, nlp, embedder, genai)
        if result:
            processed.append(result)

    return {
        "source"         : source,
        "ocr_json"       : ocr_json_path,
        "paragraph_count": len(processed),
        "paragraphs"     : processed,
        "metadata"       : {
            "timestamp"       : datetime.now().isoformat(),
            "pipeline_version": "neuro-lens-nlp-v1",
        },
    }


# ─────────────────────────────────────────────────────────────
# 11. SAVE + PRINT
# ─────────────────────────────────────────────────────────────

def save_nlp_result(result: dict, output_dir: str = "neurolens_output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"nlp_result_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[Neuro-Lens NLP] Saved → {path}")
    return path


def print_nlp_summary(result: dict):
    sep = "=" * 62
    print(f"\n{sep}")
    print("  NEURO-LENS — NLP RESULT")
    print(sep)
    print(f"  Paragraphs processed : {result['paragraph_count']}")
    print(f"  Source               : {result['source']}")
    print("-" * 62)

    for i, para in enumerate(result["paragraphs"], 1):
        cb = para["complexity_before"]
        ca = para["complexity_after"]
        print(f"\n  ── PARAGRAPH {i} ──")
        print(f"  Flesch before: {cb['flesch_ease']}  →  after: {ca['flesch_ease']}")
        print(f"  FK grade before: {cb['fk_grade']}  →  after: {ca['fk_grade']}")
        print(f"  Readability: {para.get('readability_check', {}).get('note', 'N/A')}")
        print(f"\n  BULLETS:")
        for b in para["bullets"]:
            print(f"    • {b}")
        print(f"\n  KEY IDEAS (for visual step):")
        for k in para["key_ideas"]:
            print(f"    ★ {k}")
        if para["substitutions"]:
            print(f"\n  SUBSTITUTIONS ({len(para['substitutions'])}):")
            for s in para["substitutions"][:5]:
                print(f"    {s['original']} → {s['replacement']}")
            if len(para["substitutions"]) > 5:
                print(f"    … and {len(para['substitutions']) - 5} more")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────
# 12. GUI
# ─────────────────────────────────────────────────────────────

def run_gui(nlp, embedder, genai, output_dir: str):
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, scrolledtext, messagebox
    except ImportError:
        print("[Neuro-Lens NLP] Tkinter not available. Use --input flag.")
        return

    root = tk.Tk()
    root.title("Neuro-Lens — Step 2: NLP Simplification")
    root.geometry("820x740")
    root.resizable(True, True)

    BG        = "#f5f5f0"      # off-white — reduces glare for dyslexic readers
    ACCENT    = "#2e7d32"
    MUTED     = "#444"
    TEXT      = "#1a1a1a"      # near-black for strong contrast
    TEXT_SUB  = "#3a5a3a"      # muted green for secondary info
    FONT      = ("Verdana", 10)
    FONT_B    = ("Verdana", 11, "bold")
    FONT_S    = ("Verdana", 9)
    FONT_BODY = ("Verdana", 12)          # body reading font — 12pt min
    FONT_HDR  = ("Verdana", 12, "bold")  # section headings
    MONO      = ("Consolas", 9)

    root.configure(bg=BG)

    # ── Header ───────────────────────────────────────────────
    hdr = tk.Frame(root, bg=ACCENT)
    hdr.pack(fill="x")
    tk.Label(hdr, text="Neuro-Lens  |  Step 2 — NLP Simplification",
             bg=ACCENT, fg="white", font=("Segoe UI", 12, "bold"),
             pady=11).pack(side="left", padx=16)

    body = tk.Frame(root, bg=BG, padx=18, pady=14)
    body.pack(fill="both", expand=True)

    # ── File picker ──────────────────────────────────────────
    pick_row = tk.Frame(body, bg=BG)
    pick_row.pack(fill="x", pady=(0, 8))

    path_var = tk.StringVar(value="No OCR JSON selected")
    tk.Label(pick_row, textvariable=path_var, bg=BG, fg=MUTED,
             font=FONT_S, anchor="w", wraplength=560).pack(side="left",
                                                            fill="x",
                                                            expand=True)

    def browse():
        # Auto-point to neurolens_output folder
        init_dir = os.path.join(os.getcwd(), "neurolens_output")
        if not os.path.isdir(init_dir):
            init_dir = os.getcwd()
        f = filedialog.askopenfilename(
            title="Select OCR JSON from Step 1",
            initialdir=init_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if f:
            path_var.set(f)
            status_var.set("JSON loaded — click Run NLP.")
            out_box.delete("1.0", "end")

    tk.Button(pick_row, text="Load OCR JSON", command=browse,
              bg=ACCENT, fg="white", font=FONT_B,
              relief="flat", padx=14, pady=6,
              cursor="hand2").pack(side="right")

    # ── Options row ──────────────────────────────────────────
    opt_row = tk.Frame(body, bg=BG)
    opt_row.pack(fill="x", pady=(0, 8))

    tk.Label(opt_row, text="Key ideas per paragraph:", bg=BG,
             font=FONT).pack(side="left")
    topn_var = tk.IntVar(value=3)
    tk.Spinbox(opt_row, from_=1, to=6, textvariable=topn_var,
               width=3, font=FONT).pack(side="left", padx=6)

    tk.Label(opt_row, text="   Max bullets:", bg=BG,
             font=FONT).pack(side="left")
    bullets_var = tk.IntVar(value=MAX_BULLETS)
    tk.Spinbox(opt_row, from_=2, to=10, textvariable=bullets_var,
               width=3, font=FONT).pack(side="left", padx=6)

    # ── Status ───────────────────────────────────────────────
    status_var = tk.StringVar(value="Load an OCR JSON file to begin.")
    tk.Label(body, textvariable=status_var, bg=BG, fg=MUTED,
             font=FONT_S, anchor="w").pack(fill="x", pady=(0, 4))

    # ── Notebook: Bullets | Full JSON ────────────────────────
    nb_style = ttk.Style()
    nb_style.theme_use("default")
    nb_style.configure("TNotebook",     background=BG, borderwidth=0)
    nb_style.configure("TNotebook.Tab", font=FONT_B, padding=[12, 5],
                        background="#e8f5e9")
    nb_style.map("TNotebook.Tab",
                 background=[("selected", ACCENT)],
                 foreground=[("selected", "white")])

    nb = ttk.Notebook(body)

    # Tab 1 — Simplified view (dyslexia-friendly display)
    tab_simple = tk.Frame(nb, bg=BG)
    nb.add(tab_simple, text="  Simplified Output  ")
    out_box = scrolledtext.ScrolledText(tab_simple, font=FONT_BODY,
                                        wrap="word", relief="flat",
                                        bg="#f5f5f0", fg=TEXT,
                                        spacing1=6, spacing2=4, spacing3=6,
                                        padx=20, pady=14)
    out_box.pack(fill="both", expand=True)
    # Tag definitions for rich text formatting
    out_box.tag_configure("heading",   font=FONT_HDR, foreground=ACCENT,
                           spacing1=14, spacing3=4)
    out_box.tag_configure("bullet",    font=FONT_BODY, foreground=TEXT,
                           lmargin1=28, lmargin2=44, spacing1=5, spacing3=5)
    out_box.tag_configure("divider",   font=FONT_S, foreground="#cccccc")
    out_box.tag_configure("stats_hdr", font=("Verdana", 10, "bold"),
                           foreground="#555", spacing1=18, spacing3=2)
    out_box.tag_configure("stats",     font=("Verdana", 9),
                           foreground="#666", lmargin1=16, spacing1=3)
    out_box.tag_configure("saved",     font=("Verdana", 9, "italic"),
                           foreground="#888", spacing1=10)

    # Tab 2 — Raw JSON
    tab_json = tk.Frame(nb, bg=BG)
    nb.add(tab_json, text="  Raw JSON  ")
    json_box = scrolledtext.ScrolledText(tab_json, font=MONO,
                                         wrap="none", relief="flat",
                                         bg="#f8f9fa", fg="#212529")
    json_box.pack(fill="both", expand=True)

    # ── Run + buttons (packed BEFORE notebook so always visible) ──
    def run_nlp():
        path = path_var.get()
        if path == "No OCR JSON selected" or not os.path.isfile(path):
            messagebox.showwarning("No file", "Please load an OCR JSON first.")
            return

        status_var.set("Running NLP pipeline … (30–60 s first run)")
        root.update()
        out_box.delete("1.0", "end")
        json_box.delete("1.0", "end")

        try:
            # Apply GUI options
            global MAX_BULLETS
            MAX_BULLETS = bullets_var.get()

            result   = run_pipeline(path, nlp, embedder, genai)
            saved    = save_nlp_result(result, output_dir)

            # ── Simplified tab — dyslexia-friendly layout ────
            out_box.delete("1.0", "end")
            stats_lines = []   # collect grading info for bottom section

            # ── SECTION 1: All simplified content together ───
            out_box.insert("end", "Simplified Text\n", "heading")
            out_box.insert("end", "─" * 60 + "\n", "divider")

            for i, para in enumerate(result["paragraphs"], 1):
                cb  = para["complexity_before"]
                ca  = para["complexity_after"]

                # Paragraph heading
                out_box.insert("end", f"\nSection {i}\n", "heading")

                # Key ideas as readable bullets
                for k in para["key_ideas"]:
                    out_box.insert("end", f"•  {k}\n", "bullet")

                # Extra bullet points (skip duplicates of key ideas)
                extras = [b for b in para["bullets"]
                          if b not in para["key_ideas"]]
                for b in extras:
                    out_box.insert("end", f"•  {b}\n", "bullet")

                # Collect stats for bottom
                subs = para["substitutions"]
                sub_str = ""
                if subs:
                    pairs = ", ".join(
                        f"{s['original']} → {s['replacement']}"
                        for s in subs[:5]
                    )
                    if len(subs) > 5:
                        pairs += f" (+{len(subs)-5} more)"
                    sub_str = f"  Words simplified: {pairs}"
                stats_lines.append((i, cb, ca,
                                    para.get("readability_check", {}).get("note", ""),
                                    sub_str))

            # ── SECTION 2: Grading & details at the bottom ───
            out_box.insert("end", "\n\n" + "─" * 60 + "\n", "divider")
            out_box.insert("end", "Readability & Grading Details\n", "stats_hdr")

            for (i, cb, ca, note, sub_str) in stats_lines:
                out_box.insert("end",
                    f"\n  Section {i} — "
                    f"Flesch: {cb['flesch_ease']} → {ca['flesch_ease']}   "
                    f"Grade: {cb['fk_grade']} → {ca['fk_grade']}\n",
                    "stats")
                out_box.insert("end", f"  {note}\n", "stats")
                if sub_str:
                    out_box.insert("end", f"{sub_str}\n", "stats")

            out_box.insert("end", f"\nSaved to: {saved}\n", "saved")

            # ── JSON tab ─────────────────────────────────────
            json_box.insert("end", json.dumps(result, indent=2,
                                               ensure_ascii=False))

            n_paras = result["paragraph_count"]
            status_var.set(
                f"Done — {n_paras} paragraph(s) simplified. Saved to {saved}"
            )

        except Exception as exc:
            import traceback
            err = traceback.format_exc()
            out_box.insert("end", f"ERROR: {exc}\n\n{err}")
            status_var.set(f"Error: {exc}")

    btn_row = tk.Frame(body, bg=BG)
    btn_row.pack(fill="x", pady=(8, 0))

    tk.Button(btn_row, text="Run NLP Simplification",
              command=run_nlp,
              bg="#1b5e20", fg="white", font=FONT_B,
              relief="flat", padx=20, pady=8,
              cursor="hand2").pack(side="right")

    tk.Button(btn_row, text="Clear",
              command=lambda: (out_box.delete("1.0", "end"),
                               json_box.delete("1.0", "end")),
              bg=BG, fg=MUTED, font=FONT, relief="flat",
              padx=12, pady=8, cursor="hand2").pack(side="right", padx=6)

    nb.pack(fill="both", expand=True)

    root.mainloop()


# ─────────────────────────────────────────────────────────────
# 13. ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Lens NLP — Step 2 (Simplification)"
    )
    parser.add_argument("--input",      type=str,
                        help="Path to OCR JSON from Step 1")
    parser.add_argument("--gui",        action="store_true",
                        help="Open GUI (default if --input not given)")
    parser.add_argument("--output-dir", default="neurolens_output",
                        help="Folder to save NLP JSON result")
    parser.add_argument("--top-n",      type=int, default=3,
                        help="Key ideas per paragraph (default: 3)")

    args = parser.parse_args()

    nlp, embedder = load_models()
    genai = NeuroLensGenAI(os.environ.get("GEMINI_API_KEY"))

    if args.input and os.path.isfile(args.input):
        result = run_pipeline(args.input, nlp, embedder, genai)
        print_nlp_summary(result)
        save_nlp_result(result, args.output_dir)
        print("[Neuro-Lens NLP] Done. Pass the NLP JSON to Step 3 (Visual).\n")
    else:
        run_gui(nlp, embedder,genai, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
