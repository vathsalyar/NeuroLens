"""
Neuro-Lens: Visual Generation Module (Step 3)
==============================================
Reads NLP JSON → generates comic-style illustrations → saves comic strips

WHY InferenceClient (not raw requests):
  The old raw HTTP API (api-inference.huggingface.co) is deprecated.
  The correct 2025 approach is huggingface_hub.InferenceClient with
  the text_to_image() method — this is what HF officially documents and
  actively maintains.

MODEL: stabilityai/stable-diffusion-2-1
  - Confirmed working on HF free inference tier in 2025
  - Produces clean illustrations suitable for educational content
  - Free with a HF account token (read permission is enough)
  - No local GPU needed — runs on HF servers

Install:
    pip install -r requirements_visual.txt

Get your free HF token:
    1. Sign up at https://huggingface.co (free)
    2. Go to https://huggingface.co/settings/tokens
    3. Click "New token" → Name it → Access: Read → Generate
    4. Copy the token (starts with hf_...)
    5. Set it:  Windows: set HF_TOKEN=hf_...
               Mac/Linux: export HF_TOKEN=hf_...
    OR just paste it in the GUI when prompted.

Run:
    python neurolens_visual.py                    # opens GUI
    python neurolens_visual.py --input neurolens_output/nlp_result_*.json
"""

import os
import io
import json
import time
import textwrap
import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import InferenceClient


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Working on HF free Inference API (2026)
PRIMARY_MODEL  = "black-forest-labs/FLUX.1-schnell"
FALLBACK_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

COMIC_DIR      = "neurolens_output/comics"

# Dyslexia-friendly style: vivid scene illustration, NO AI-generated text
STYLE_SUFFIX = (
    "flat vector illustration, comic book panel style, clean bold outlines, "
    "bright saturated colors, children's science textbook art, "
    "white background, clear vivid scene, no text, no letters, no labels, "
    "no arrows, no words, no numbers, no writing of any kind"
)

NEGATIVE_PROMPT = (
    "text, letters, words, labels, numbers, arrows, watermark, signature, "
    "writing, typography, fonts, captions, annotations, logo, blurry, "
    "dark, scary, violent, horror, realistic photo, photograph, 3d render, "
    "adult content, nsfw, ugly, low quality, deformed, abstract"
)

# Image size — square panels for comic grid
IMG_SIZE = 512

# Panel layout
PANEL_PAD    = 10
CAPTION_H    = 80
BORDER_W     = 3
COLS         = 2          # panels per row in strip
TITLE_H      = 55

# Colors
C_WHITE      = (255, 255, 255)
C_BORDER     = (40,  40,  40)
C_CAP_BG     = (248, 248, 242)
C_CAP_TEXT   = (25,  25,  25)
C_TITLE_BG   = (30,  90, 200)
C_TITLE_TEXT = (255, 255, 255)
C_IDEA_BG    = (230, 240, 255)
C_STAR       = (30,  90, 200)


# ─────────────────────────────────────────────────────────────
# CONCEPT MAP — key concept → safe visual prompt
# ─────────────────────────────────────────────────────────────
CONCEPT_MAP = {
    # Biology / Science
    "photosynthesis" : "green plant absorbing sunlight with glowing leaves",
    "cell"           : "colorful round cell with nucleus inside",
    "mitosis"        : "one cell dividing into two identical cells",
    "dna"            : "colorful double helix spiral strand",
    "respiration"    : "cartoon lungs breathing in clean air",
    "digestion"      : "cartoon stomach with food being broken down",
    "heart"          : "cartoon heart pumping with arrows showing blood flow",
    "brain"          : "cartoon brain with glowing light bulb neurons",
    "atom"           : "colorful atom with electrons orbiting nucleus",
    "molecule"       : "three colorful spheres bonded together",
    "gravity"        : "red apple falling down from a green tree",
    "electricity"    : "yellow lightning bolt connecting battery to light bulb",
    "magnet"         : "blue horseshoe magnet attracting small iron pieces",
    "water cycle"    : "sun evaporating water, cloud forming, rain falling",
    "ecosystem"      : "sun shining on plant, rabbit eating plant, fox watching",
    "volcano"        : "triangular mountain erupting with orange lava",
    "earthquake"     : "ground cracking open with zigzag lines",
    "planet"         : "colorful solar system with eight planets in orbit",
    "chlorophyll"    : "green leaf close-up with bright sun above",
    "surface area"   : "wide flat green leaf showing large surface catching sunlight",
    "carbon dioxide" : "factory with arrows going into green plant",
    "dioxide"        : "factory with arrows going into green plant",
    "glucose"        : "glowing glucose molecule made by green plant",
    "leaves"         : "single green leaf with visible veins close-up",
    "cells"          : "colorful round cell with nucleus inside",
    "oxygen"         : "green plant releasing small bubbles upward",
    "carbon dioxide" : "factory with arrows going into green plant",
    "energy"         : "bright yellow sun with radiating energy arrows",
    "sunlight"       : "bright yellow sun shining on green earth below",
    "root"           : "tree with visible roots absorbing water underground",
    "leaf"           : "single green leaf with visible veins close-up",
    "food"           : "colorful plate with fruits vegetables and grains",
    "sugar"          : "glowing glucose molecule made by green plant",
    "life"           : "diverse animals and plants together in nature",
    "air"            : "wind blowing through tree leaves with motion lines",
    "water"          : "clear blue water droplet splashing on surface",
    # Math / Physics
    "fraction"       : "colorful pizza divided into eight equal slices",
    "geometry"       : "colorful triangle circle and square side by side",
    "equation"       : "balance scale with numbers on each side",
    "force"          : "red arrow showing a push on a blue box",
    "wave"           : "blue sine wave on a white background",
    "light"          : "prism splitting white light into rainbow",
    "temperature"    : "red thermometer showing hot versus cold",
    "speed"          : "cartoon racing car with motion blur lines",
    # History / Social
    "democracy"      : "people putting ballot papers in a box",
    "trade"          : "two people exchanging colorful packages",
    "farming"        : "farmer in green field planting colorful crops",
    "communication"  : "two people with colorful speech bubbles between them",
    # General / Education
    "learn"          : "child reading open book with lightbulb above head",
    "learning"       : "child reading open book with lightbulb above head",
    "school"         : "red brick schoolhouse with green trees beside it",
    "read"           : "person happily reading an open colorful book",
    "reading"        : "person happily reading an open colorful book",
    "write"          : "hand holding pencil writing on white paper",
    "memory"         : "cartoon head with visible gears and lightbulb inside",
    "think"          : "cartoon head with colorful thought bubble floating up",
    "idea"           : "glowing yellow lightbulb above a smiling cartoon head",
    "question"       : "large colorful question mark with curious face",
    "answer"         : "green checkmark next to smiling cartoon face",
    "problem"        : "tangled colorful rope knot being slowly untied",
    "solution"       : "golden key unlocking a blue padlock",
    "group"          : "three smiling cartoon people standing together",
    "time"           : "cartoon clock face showing hands moving forward",
    "growth"         : "small seed sprouting into tall green tree step by step",
    "change"         : "caterpillar in cocoon transforming into butterfly",
    "cycle"          : "four-step circular arrow diagram with colorful steps",
    "system"         : "three colorful interlocking gears turning together",
    "data"           : "colorful bar chart with different height bars",
    "computer"       : "friendly cartoon desktop computer with smile face screen",
    "network"        : "colorful dots connected by lines forming a web",
    "number"         : "large colorful numbers floating in blue sky",
    "pattern"        : "repeating row of colorful stars circles and squares",
    "body"           : "simple outline of smiling human body labeled parts",
    "earth"          : "blue and green cartoon globe with smile face",
    "sun"            : "bright yellow cartoon sun with triangular rays smiling",
    "moon"           : "silver crescent moon with stars on dark blue background",
    "animal"         : "colorful friendly cartoon animals lion elephant giraffe",
    "plant"          : "cheerful green plant with many leaves in orange pot",
    "tree"           : "tall green tree with apples on branches",
    "flower"         : "colorful sunflower with yellow petals and brown center",
    "ocean"          : "blue ocean waves with fish and coral visible",
    "mountain"       : "snow-capped triangular mountain with blue sky",
    "city"           : "colorful cartoon city skyline with tall buildings",
    "family"         : "cartoon family parents and children holding hands",
    "health"         : "person running with green heart symbol above",
    "help"           : "one hand reaching to help another hand up",
    "work"           : "person at desk using tools and computer",
    "play"           : "child playing happily with colorful toy blocks",
    "music"          : "colorful musical notes floating from violin",
    "sport"          : "cartoon person running fast on green track",
    "push"           : "arrow showing force pushing an object forward",
    "pull"           : "arrow showing force pulling an object toward you",
}

def split_into_panels(text, max_panels=3):
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences[:max_panels]

# ─────────────────────────────────────────────────────────────
# 1. PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def build_prompt(key_idea: str, original: str = "") -> str:
    """
    Build a rich, scene-specific image prompt using the ORIGINAL sentence.

    Strategy:
    1. Always prefer the original (unmangled) sentence as the content source
    2. Build a scene description: WHO is doing WHAT with WHICH objects
    3. Map specific biology/science vocabulary to visual scene elements
    4. Include multiple labeled scene elements so the image conveys meaning
    5. Add dyslexia-friendly comic style suffix
    """
    # Always use original text when available — simplified text is mangled
    source = original.strip() if original.strip() else key_idea.strip()
    src_lower = source.lower()

    # ── Scene templates keyed on dominant concept in the original text ──
    # Each entry: (trigger_keywords, scene_description)
    # More specific patterns are listed first (checked in order)
    SCENE_TEMPLATES = [
        # ── Most-specific matches FIRST ──────────────────────────────

        # P2-I3: Calvin cycle converts CO2 → glucose (most specific: "convert carbon dioxide into")
        (["convert carbon dioxide into", "carbon dioxide into glucose"],
         "a large golden gear wheel spins in center on a white background: "
         "at the top of the gear small grey round puff shapes feed in; "
         "inside the gear blue-green molecular shapes swirl in a circle; "
         "at the bottom bright golden hexagonal crystal chunks pop out; "
         "small glowing green spark shapes float around the outside of the gear; "
         "bright vivid flat illustration, candy colors, no text"),

        # P2-I2: light-dependent reactions producing ATP/NADPH
        (["capture sunlight", "energy molecules", "atp and nadph",
          "atp", "nadph"],
         "a glowing green chloroplast floats in center of frame, "
         "bright yellow sunbeams strike its left surface and transform into "
         "small glowing blue energy orbs (ATP) and pink orbs (NADPH) that "
         "float away to the right; the chloroplast surface ripples with golden "
         "light where the beams hit; vivid flat cartoon, no writing"),

        # P3-I2: CO2 balance in the atmosphere
        (["atmosphere", "balance", "maintain", "helps maintain"],
         "a calm nature scene: on the left a dense green forest with trees "
         "whose leaves are bright green, grey wisps flow from the sky toward "
         "the trees; on the right a clear blue sky with fluffy white clouds; "
         "a large old-fashioned balance scale sits in the foreground, both "
         "pans perfectly level, one holds a tiny leafy green plant and the "
         "other a small grey cloud puff; peaceful bright flat illustration"),

        # P2-I1: two main stages overview
        (["calvin cycle", "light-dependent", "atp", "nadph",
          "two main stages"],
         "inside a giant green leaf cross-section: on the left half "
         "bright yellow sun rays shine down onto stacked green disc shapes "
         "(thylakoids) inside a green oval chloroplast, glowing energy dots "
         "flow out from them; on the right half a circular wheel of "
         "molecular shapes spins, small hexagonal sugar crystals form at the "
         "bottom of the wheel; connecting glow arrows between left and right"),

        (["light-dependent", "capture sunlight", "energy molecules",
          "atp", "nadph"],
         "a glowing green chloroplast floats in center of frame, "
         "bright yellow sunbeams hit the left side of it and transform into "
         "small glowing blue energy orbs floating away to the right, "
         "the chloroplast glows warm yellow-green where sunlight touches it, "
         "vivid flat illustration, no writing"),

        (["convert carbon dioxide into", "carbon dioxide into glucose",
          "used in the calvin"],
         "a large circular wheel drawn like a shining golden gear: "
         "small grey round puff shapes feed in at the top of the gear; "
         "inside the gear green and blue spark shapes swirl; "
         "at the bottom bright golden hexagonal crystal shapes pop out "
         "representing glucose sugar; surrounding the gear are small "
         "glowing green spark shapes; vivid flat candy-colored illustration"),

        # ── Chloroplast / chlorophyll ────────────────────────────────
        (["chloroplast", "chlorophyll", "absorbs light", "light energy",
          "takes place in"],
         "a single large green oval plant cell, inside it three bright "
         "emerald-green lens-shaped chloroplasts are highlighted with a "
         "glowing outline, golden sunbeam rays enter from top-left and hit "
         "the chloroplasts which light up bright green, small energy sparks "
         "radiate outward from each chloroplast, clean vivid flat art"),

        # ── Inputs and outputs of photosynthesis ────────────────────
        (["food chain", "vital for life", "base of", "forms the base",
          "produces oxygen", "food chain"],
         "a bright sunny meadow scene: a large green leafy tree in center "
         "with white bubbles floating upward from its leaves into the blue sky; "
         "to the left a rabbit nibbles grass, to the right a fox watches; "
         "above the scene the sun shines with rays; at the bottom lush green "
         "grass and flowers grow; the whole scene feels alive and thriving; "
         "vivid flat cartoon illustration"),

        # ── Who does photosynthesis ──────────────────────────────────
        (["green plants", "algae", "bacteria", "sunlight",
          "make their own food", "own food"],
         "three side-by-side panels on a white background: "
         "left panel shows a tall green leafy plant in bright sunlight; "
         "middle panel shows blue-green wavy water with small round algae "
         "cells floating and glowing under a sun; "
         "right panel shows tiny rod-shaped bacteria near a glowing light; "
         "all three have a small golden bowl of food appearing beside them, "
         "bright cartoon flat illustration"),

        # ── Importance / food chain / oxygen ────────────────────────
        (["carbon dioxide", "water", "soil", "glucose", "oxygen",
          "produce", "release"],
         "a cheerful green plant with visible roots in brown soil: "
         "from the left blue wavy lines (water) travel up through the roots "
         "and stem; from above a bright sun shines yellow rays onto the "
         "broad flat leaves; small grey cloud puffs drift toward the leaves; "
         "inside the leaves tiny golden hexagon shapes (glucose) glow; "
         "white round bubbles float upward from the leaves into the blue sky; "
         "clean flat vector, no writing"),

        # ── CO2 balance / atmosphere ────────────────────────────────
        (["atmosphere", "balance", "carbon dioxide", "maintain",
          "helps maintain"],
         "a calm nature scene: on one side a dense green forest with trees "
         "whose leaves are bright green, small grey wisps flow from the sky "
         "toward the trees; on the other side a clear blue sky with fluffy "
         "white clouds; a large old-fashioned balance scale sits in the "
         "foreground, both pans are level, one pan holds a tiny leafy plant "
         "and the other holds a small grey cloud puff; "
         "peaceful bright flat illustration, no text"),

        # ── Without photosynthesis ───────────────────────────────────
        (["without", "survive", "living organisms", "would not",
          "most living"],
         "a split scene divided by a bold vertical line: "
         "left half shows a lush vibrant world — tall green trees, colorful "
         "flowers, blue sky, a rabbit, a bird, a happy human silhouette; "
         "right half shows the same scene but lifeless — grey bare trees, "
         "cracked brown earth, grey sky, wilted plants, all desaturated; "
         "bold clean flat illustration, vivid contrast between the two halves"),

        # ── General photosynthesis overview ─────────────────────────
        (["photosynthesis", "process", "green plant"],
         "a large friendly green plant with big flat leaves stands in the "
         "center: bright yellow sun above shines rays onto the leaves; "
         "blue wavy water rises up through the brown roots and green stem; "
         "small grey cloud puff near the leaves representing air; "
         "inside the leaves a warm golden glow; white round bubbles float "
         "up from the leaves into the clear blue sky above; "
         "cheerful vivid flat cartoon, clean white background"),
    ]

    for triggers, scene in SCENE_TEMPLATES:
        if any(t in src_lower for t in triggers):
            return f"{scene}, {STYLE_SUFFIX}"

    # ── Fallback: build scene from key nouns in original text ──────────
    STOP = {"the","a","an","is","are","was","were","be","been","being","have",
            "has","had","do","does","did","will","would","could","should","may",
            "might","can","shall","this","that","these","those","it","its",
            "they","them","their","we","our","you","your","he","she","his",
            "her","which","who","what","when","where","how","why","and","or",
            "but","if","so","as","at","by","for","of","on","to","up","in",
            "out","not","with","into","through","also","then","than","some",
            "most","all","both","each","other","such","very","just","more"}
    words = source.replace(",","").replace(".","").lower().split()
    core  = [w for w in words if w not in STOP and len(w) > 3 and w.isalpha()]
    if core:
        subject = " ".join(core[:5])
        return (f"detailed educational diagram illustrating {subject}, "
                f"labeled elements with arrows, clear scene, {STYLE_SUFFIX}")

    return f"educational science diagram with labeled elements, {STYLE_SUFFIX}"


# ─────────────────────────────────────────────────────────────
# 2. IMAGE GENERATION  (huggingface_hub InferenceClient)
# ─────────────────────────────────────────────────────────────

def make_client(hf_token: str) -> InferenceClient:
    return InferenceClient(token=hf_token)


def generate_image(prompt: str, client: InferenceClient,
                   model: str = PRIMARY_MODEL) -> Image.Image | None:
    """
    Generate one image using HF InferenceClient.text_to_image().
    Returns PIL Image or None on failure.
    Tries fallback model if primary fails.
    """
    is_flux = "flux" in model.lower()
    steps   = 4 if is_flux else 25
    extra   = {} if is_flux else {"negative_prompt": NEGATIVE_PROMPT,
                                   "guidance_scale": 7.5}

    for attempt in range(1, 4):
        try:
            print(f"    [Visual] Calling {model} (attempt {attempt}) …")
            image = client.text_to_image(
                prompt,
                model=model,
                num_inference_steps=steps,
                width=IMG_SIZE,
                height=IMG_SIZE,
                **extra,
            )
            print(f"    [Visual] Image generated successfully.")
            return image

        except Exception as exc:
            err = str(exc)
            print(f"    [Visual] Attempt {attempt} failed: {err[:120]}")

            if "404" in err:
                print(f"    [Visual] Model {model} not found on HF Inference API.")
                if model == PRIMARY_MODEL:
                    print(f"    [Visual] Trying fallback: {FALLBACK_MODEL}")
                    return generate_image(prompt, client, FALLBACK_MODEL)
                return None

            elif "loading" in err.lower() or "503" in err:
                wait = 20 * attempt
                print(f"    [Visual] Model loading — waiting {wait}s …")
                time.sleep(wait)

            elif "402" in err or "credit" in err.lower():
                print("    [Visual] Monthly free credits exhausted.")
                print("             Upgrade to HF PRO or wait until next month.")
                return None

            elif "401" in err or "unauthorized" in err.lower():
                print("    [Visual] Invalid HF token — check your token.")
                return None

            elif attempt == 3 and model == PRIMARY_MODEL:
                print(f"    [Visual] Trying fallback: {FALLBACK_MODEL}")
                return generate_image(prompt, client, FALLBACK_MODEL)

    return None


# ─────────────────────────────────────────────────────────────
# 3. COMIC PANEL BUILDER
# ─────────────────────────────────────────────────────────────

def _font(size: int) -> ImageFont.ImageFont:
    """Load a clean readable font with fallback to default."""
    candidates = [
        "arial.ttf", "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def draw_dialogue_bubble(draw: ImageDraw.ImageDraw, text: str,
                          x: int, y: int, w: int, font,
                          bg=(255, 255, 240), border=(50, 50, 180)):
    """
    Draw a rounded dialogue bubble containing text at position (x,y).
    Returns the height used.
    """
    pad   = 10
    lines = textwrap.wrap(text, width=max(24, w // 8))
    lh    = 18   # line height in pixels
    bh    = len(lines) * lh + 2 * pad
    bw    = w

    # Bubble background + border
    draw.rounded_rectangle([x, y, x + bw, y + bh],
                            radius=12, fill=bg, outline=border, width=2)

    # Tail triangle pointing downward-left
    tail = [(x + 24, y + bh),
            (x + 14, y + bh + 14),
            (x + 40, y + bh)]
    draw.polygon(tail, fill=bg, outline=border)

    # Text inside bubble
    for i, line in enumerate(lines):
        draw.text((x + pad, y + pad + i * lh), line,
                  fill=(20, 20, 80), font=font)
    return bh + 16   # total height including tail


def make_panel(image: Image.Image, caption: str, number: int,
               dialogue_text: str = "") -> Image.Image:
    """
    Assemble one comic panel:
    - Generated image fills top portion
    - Dialogue bubble overlaid on image bottom with the KEY IDEA text
    - Caption bar below with the original sentence (full text)
    Dyslexia design: Verdana-style fonts, high contrast, generous spacing.
    """
    BUBBLE_H   = 110   # reserved space at image bottom for bubble
    CAP_HEIGHT = 90    # caption bar height
    pw = IMG_SIZE + 2 * PANEL_PAD
    ph = IMG_SIZE + CAP_HEIGHT + 3 * PANEL_PAD

    panel = Image.new("RGB", (pw, ph), C_WHITE)
    draw  = ImageDraw.Draw(panel)

    # Outer border
    draw.rectangle([0, 0, pw - 1, ph - 1], outline=C_BORDER, width=BORDER_W)

    # Paste image — slightly shorter to leave room at bottom for bubble
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    panel.paste(img_resized, (PANEL_PAD, PANEL_PAD))

    # ── Dialogue bubble overlaid on bottom of image ──────────────────
    bubble_text = dialogue_text if dialogue_text else caption
    # Keep bubble text concise — first 120 chars max
    bubble_text = bubble_text[:200].rsplit(" ", 1)[0] if len(bubble_text) > 200 else bubble_text
    fb = _font(14)
    bub_w   = IMG_SIZE - 20
    bub_x   = PANEL_PAD + 10
    bub_y   = PANEL_PAD + IMG_SIZE - BUBBLE_H - 10
    draw_dialogue_bubble(draw, bubble_text, bub_x, bub_y, bub_w, fb,
                         bg=(255, 255, 220), border=(30, 80, 200))

    # ── Caption bar below image ───────────────────────────────────────
    cap_y = PANEL_PAD + IMG_SIZE + PANEL_PAD // 2
    draw.rectangle([BORDER_W, cap_y, pw - BORDER_W - 1, ph - BORDER_W - 1],
                   fill=C_CAP_BG)

    # Panel number badge
    fn = _font(12)
    draw.text((BORDER_W + 8, cap_y + 4), f"#{number}",
              fill=C_STAR, font=fn)

    # Full original caption text — large, readable, wrapped
    fc = _font(13)
    wrapped = textwrap.fill(caption, width=52)
    draw.text((BORDER_W + 8, cap_y + 22), wrapped,
              fill=C_CAP_TEXT, font=fc)

    return panel


def make_placeholder(caption: str, number: int) -> Image.Image:
    """
    Placeholder panel when image generation fails.
    Shows the key idea text prominently so learning is not blocked.
    """
    pw = IMG_SIZE + 2 * PANEL_PAD
    ph = IMG_SIZE + CAPTION_H + 3 * PANEL_PAD

    panel = Image.new("RGB", (pw, ph), (235, 245, 255))
    draw  = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, pw - 1, ph - 1], outline=C_BORDER, width=BORDER_W)

    # Icon area
    draw.rectangle([PANEL_PAD, PANEL_PAD,
                    pw - PANEL_PAD, PANEL_PAD + IMG_SIZE],
                   fill=(210, 228, 255), outline=(170, 200, 240))

    # Book symbol
    bx, by = pw // 2, PANEL_PAD + IMG_SIZE // 2
    draw.rectangle([bx - 60, by - 70, bx + 60, by + 70],
                   fill=(255, 255, 255), outline=C_BORDER, width=2)
    draw.line([bx, by - 70, bx, by + 70], fill=C_BORDER, width=2)
    for ly in range(by - 50, by + 50, 12):
        draw.line([bx + 8, ly, bx + 52, ly],
                  fill=(180, 200, 230), width=1)

    # "Key Idea" label inside panel
    fi = _font(15)
    draw.text((PANEL_PAD + 8, PANEL_PAD + 12), "Key Idea:",
              fill=C_STAR, font=fi)
    fw = _font(13)
    idea_wrapped = textwrap.fill(caption, width=44)
    draw.text((PANEL_PAD + 8, PANEL_PAD + 34), idea_wrapped,
              fill=(30, 30, 80), font=fw)

    # Caption
    cap_y = PANEL_PAD + IMG_SIZE + PANEL_PAD // 2
    draw.rectangle([BORDER_W, cap_y, pw - BORDER_W - 1, ph - BORDER_W - 1],
                   fill=C_CAP_BG)
    fn = _font(12)
    draw.text((BORDER_W + 8, cap_y + 5), f"#{number}",
              fill=C_STAR, font=fn)
    fc = _font(13)
    draw.text((BORDER_W + 8, cap_y + 20),
              textwrap.fill(caption, width=50),
              fill=C_CAP_TEXT, font=fc)

    return panel


def assemble_strip(panels: list, title: str) -> Image.Image:
    """
    Arrange panels into a grid comic strip with a title banner.
    """
    pw     = panels[0].width
    ph     = panels[0].height
    n_rows = (len(panels) + COLS - 1) // COLS

    sw = pw * COLS + PANEL_PAD * (COLS + 1)
    sh = ph * n_rows + PANEL_PAD * (n_rows + 1) + TITLE_H

    strip = Image.new("RGB", (sw, sh), (252, 252, 250))
    draw  = ImageDraw.Draw(strip)

    # Title banner
    draw.rectangle([0, 0, sw, TITLE_H], fill=C_TITLE_BG)
    ft = _font(20)
    draw.text((16, 15), title, fill=C_TITLE_TEXT, font=ft)

    # Paste panels
    for i, panel in enumerate(panels):
        row = i // COLS
        col = i %  COLS
        x   = PANEL_PAD + col * (pw + PANEL_PAD)
        y   = TITLE_H + PANEL_PAD + row * (ph + PANEL_PAD)
        strip.paste(panel, (x, y))

    return strip


# ─────────────────────────────────────────────────────────────
# 4. PARAGRAPH VISUAL PIPELINE
# ─────────────────────────────────────────────────────────────

def process_paragraph(para: dict, para_idx: int,
                       client: InferenceClient,
                       out_dir: str) -> dict:
    """
    For one NLP paragraph: generate a panel per key_idea → assemble strip.
    """
    key_ideas = para.get("key_ideas", [])
    if not key_ideas:
        print(f"  [Visual] Para {para_idx}: no key ideas, skipping.")
        return {}

    os.makedirs(out_dir, exist_ok=True)

    panels     = []
    panel_meta = []
        # --- Comic Panel Generation ---
    genai_text = para.get("genai_output", {}).get("explanation", "")

    if genai_text:
        panel_texts = split_into_panels(genai_text)
    else:
        panel_texts = key_ideas

    for idx, idea in enumerate(panel_texts, 1):
        print(f"\n  [Visual] Para {para_idx} | Panel {idx}/{len(panel_texts)}")
        print(f"    Idea: {idea[:80]}")

        original_text = para.get("original", "")
        # --- GenAI Integration ---
        genai_text = para.get("genai_output", {}).get("explanation", "")
        prompt = build_prompt(idea, original=original_text)
        # Enhance prompt using GenAI
        if genai_text:
            prompt += f", scene showing: {idea}"        
        print(f"    Prompt: {prompt[:100]}")

        img = generate_image(prompt, client)

        # Use the ORIGINAL sentence as caption (not the mangled simplified one)
        original_text = para.get("original", "")
        # Split original into sentences and pick the one most matching this idea
        orig_sents = [s.strip() for s in original_text.replace("?",".")
                      .replace("!",".").split(".") if len(s.strip()) > 10]

        # Find best matching original sentence for this idea (by shared words)
        idea_words_set = set(idea.lower().split())
        best_orig = original_text  # fallback: full original
        best_score = 0
        for sent in orig_sents:
            score = len(set(sent.lower().split()) & idea_words_set)
            if score > best_score:
                best_score = score
                best_orig = sent

        caption  = best_orig[:120] if best_orig else (idea[:78] + "…" if len(idea) > 80 else idea)
        # Dialogue bubble always shows the ORIGINAL clear sentence
        dialogue = idea[:160]
        if img is not None:
            panel      = make_panel(img, caption, idx, dialogue_text=dialogue)
            panel_name = f"para{para_idx:02d}_idea{idx:02d}.png"
            panel_path = os.path.join(out_dir, panel_name)
            panel.save(panel_path)
            print(f"    Saved → {panel_path}")
        else:
            panel      = make_placeholder(caption, idx)
            panel_path = None
            print(f"    Using placeholder panel.")

        panels.append(panel)
        panel_meta.append({
            "idea"      : idea,
            "prompt"    : prompt,
            "panel_path": panel_path,
            "generated" : img is not None,
        })

        # Small delay between API calls to stay within rate limits
        if idx < len(panel_texts):
            time.sleep(4)

    # Assemble strip
    title      = f"Section {para_idx} — Key Ideas"
    strip      = assemble_strip(panels, title)
    strip_name = f"comic_strip_para{para_idx:02d}.png"
    strip_path = os.path.join(out_dir, strip_name)
    strip.save(strip_path)
    print(f"\n  [Visual] Strip saved → {strip_path}")

    return {
        "paragraph_index": para_idx,
        "strip_path"     : strip_path,
        "panels"         : panel_meta,
        "key_ideas_count": len(key_ideas),
        "generated_count": sum(1 for p in panel_meta if p["generated"]),
    }



# ─────────────────────────────────────────────────────────────
# PARAGRAPH CONSOLIDATION
# ─────────────────────────────────────────────────────────────

def consolidate_paragraphs(paragraphs: list) -> list:
    """
    OCR often splits one logical paragraph into many short fragments
    (e.g. "is called photosynthesis.", "Their", "way to make.").
    This function merges those fragments into coherent visual sections
    and produces clean, meaningful key_ideas for image generation.

    Rules:
    - A paragraph is a "stub" if its original text has < 6 words OR
      its key_ideas are all < 4 meaningful content words each.
    - Stubs are merged with their neighbours into a group.
    - Each group becomes one visual section with consolidated ideas.
    - key_ideas are filtered and de-duped; stubs within a group
      are dropped; the group's longest/richest ideas are kept (max 3).
    - original text is also merged so build_prompt can use it.
    """
    STOP = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","can","shall","this","that","these",
        "those","it","its","they","them","their","we","our","you",
        "your","he","she","his","her","which","who","what","when",
        "where","how","why","and","or","but","if","so","as","at",
        "by","for","of","on","to","up","in","out","not","with",
        "name","front","step","way","form","supply","stage","wide",
        "flat","big","well","fit","due","absorbed","special",
    }

    def meaningful_words(text):
        words = text.replace(",","").replace(".","").lower().split()
        return [w for w in words if w not in STOP and len(w) > 3 and w.isalpha()]

    def is_stub(para):
        orig_words = para.get("original","").split()
        if len(orig_words) < 6:
            return True
        ideas = para.get("key_ideas", [])
        if not ideas:
            return True
        # All ideas are stubs if every one has < 4 meaningful words
        return all(len(meaningful_words(i)) < 4 for i in ideas)

    def is_valid_idea(idea):
        """An idea is valid if it has ≥ 3 meaningful content words."""
        return len(meaningful_words(idea)) >= 3

    # ── Group paragraphs: merge stubs into surrounding real paragraphs ─
    groups = []
    current_group = []

    for para in paragraphs:
        current_group.append(para)
        if not is_stub(para):
            # This is a real paragraph — close the group
            groups.append(current_group)
            current_group = []

    # Any trailing stubs get added to last group or their own group
    if current_group:
        if groups:
            groups[-1].extend(current_group)
        else:
            groups.append(current_group)

    # ── Build consolidated paragraphs ────────────────────────────────
    consolidated = []
    for group in groups:
        # Merge original text
        merged_original = " ".join(
            p.get("original", "") for p in group
        ).strip()

        # Collect all key_ideas, filter stubs, de-dup
        all_ideas = []
        seen = set()
        for p in group:
            for idea in p.get("key_ideas", []):
                idea_clean = idea.strip().rstrip(".")
                if is_valid_idea(idea_clean) and idea_clean.lower() not in seen:
                    seen.add(idea_clean.lower())
                    all_ideas.append(idea_clean)

        # If no valid ideas, use the merged original sentences split by period
        if not all_ideas:
            for sent in merged_original.split("."):
                sent = sent.strip()
                if is_valid_idea(sent):
                    all_ideas.append(sent)

        # Cap at 3 ideas — pick the richest (most meaningful words)
        all_ideas.sort(key=lambda x: len(meaningful_words(x)), reverse=True)
        all_ideas = all_ideas[:3]

        # Fallback: use merged original as the single idea
        if not all_ideas:
            all_ideas = [merged_original[:120]]

        # Take the best readability check from group
        rc = next(
            (p.get("readability_check") for p in group
             if p.get("readability_check", {}).get("passed")),
            group[-1].get("readability_check", {})
        )

        consolidated.append({
            "original"        : merged_original,
            "key_ideas"       : all_ideas,
            "bullets"         : [p.get("bullets", []) for p in group],
            "readability_check": rc,
            "_source_count"   : len(group),
        })

    print(f"[Visual] Consolidated {len(paragraphs)} OCR fragments → "
          f"{len(consolidated)} visual section(s).")
    return consolidated

# ─────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_visual_pipeline(nlp_json: str, hf_token: str,
                         out_dir: str = COMIC_DIR) -> dict:
    with open(nlp_json, encoding="utf-8") as f:
        nlp_data = json.load(f)

    paragraphs = nlp_data.get("paragraphs", [])
    source     = nlp_data.get("source", nlp_json)

    # Merge OCR fragments into coherent visual sections
    paragraphs = consolidate_paragraphs(paragraphs)

    print(f"\n[Neuro-Lens Visual] {len(paragraphs)} paragraph(s) to illustrate")
    print(f"[Neuro-Lens Visual] Model: {PRIMARY_MODEL}")
    print(f"[Neuro-Lens Visual] Output dir: {out_dir}\n")

    client  = make_client(hf_token)
    results = []

    for idx, para in enumerate(paragraphs, 1):
        r = process_paragraph(para, idx, client, out_dir)
        if r:
            results.append(r)

    total_gen = sum(p["generated_count"] for p in results
                    if "generated_count" in p)
    total_panels = sum(p["key_ideas_count"] for p in results
                       if "key_ideas_count" in p)

    output = {
        "source"          : source,
        "nlp_json"        : nlp_json,
        "total_strips"    : len(results),
        "total_panels"    : total_panels,
        "generated_panels": total_gen,
        "output_dir"      : out_dir,
        "visual_results"  : results,
        "metadata": {
            "timestamp" : datetime.now().isoformat(),
            "model_used": PRIMARY_MODEL,
        }
    }

    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    jout = os.path.join(out_dir, f"visual_result_{ts}.json")
    with open(jout, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Neuro-Lens Visual] Result JSON → {jout}")
    print(f"[Neuro-Lens Visual] Generated {total_gen}/{total_panels} images.")

    return output


# ─────────────────────────────────────────────────────────────
# 6. GUI
# ─────────────────────────────────────────────────────────────

def run_gui(out_dir: str):
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, scrolledtext, messagebox
    except ImportError:
        print("Tkinter not available. Use --input flag.")
        return

    root = tk.Tk()
    root.title("Neuro-Lens — Step 3: Comic Generation")
    root.geometry("820x700")
    root.resizable(True, True)

    BG     = "#ffffff"
    ACCENT = "#7b1fa2"
    MUTED  = "#555"
    FONT   = ("Segoe UI", 10)
    FONT_B = ("Segoe UI", 10, "bold")
    FONT_S = ("Segoe UI", 9)
    MONO   = ("Consolas", 9)

    root.configure(bg=BG)

    # Header
    hdr = tk.Frame(root, bg=ACCENT)
    hdr.pack(fill="x")
    tk.Label(hdr, text="Neuro-Lens  |  Step 3 — Comic Visual Generation",
             bg=ACCENT, fg="white",
             font=("Segoe UI", 12, "bold"), pady=11).pack(side="left", padx=16)

    body = tk.Frame(root, bg=BG, padx=18, pady=14)
    body.pack(fill="both", expand=True)

    # ── JSON picker
    r1 = tk.Frame(body, bg=BG)
    r1.pack(fill="x", pady=(0, 8))

    path_var = tk.StringVar(value="No NLP JSON selected")
    tk.Label(r1, textvariable=path_var, bg=BG, fg=MUTED,
             font=FONT_S, anchor="w", wraplength=560).pack(side="left",
                                                            fill="x",
                                                            expand=True)

    def browse():
        init = os.path.join(os.getcwd(), "neurolens_output")
        if not os.path.isdir(init):
            init = os.getcwd()
        f = filedialog.askopenfilename(
            title="Select NLP result JSON (Step 2 output)",
            initialdir=init,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if f:
            path_var.set(f)
            status_var.set("JSON loaded — enter your HF token and click Generate.")
            log.delete("1.0", "end")

    tk.Button(r1, text="Load NLP JSON", command=browse,
              bg=ACCENT, fg="white", font=FONT_B,
              relief="flat", padx=14, pady=6,
              cursor="hand2").pack(side="right")

    # ── Token row
    r2 = tk.Frame(body, bg=BG)
    r2.pack(fill="x", pady=(0, 4))

    tk.Label(r2, text="HF Token:", bg=BG, font=FONT).pack(side="left")
    tok_var  = tk.StringVar(value=os.environ.get("HF_TOKEN", ""))
    tok_ent  = tk.Entry(r2, textvariable=tok_var, width=44, font=FONT, show="*")
    tok_ent.pack(side="left", padx=6)

    show_v = tk.BooleanVar()
    tk.Checkbutton(r2, text="Show", variable=show_v, bg=BG, font=FONT_S,
                   command=lambda: tok_ent.config(
                       show="" if show_v.get() else "*")
                   ).pack(side="left")

    # ── Token help
    tk.Label(body,
             text="No token? → huggingface.co/settings/tokens  "
                  "(free account, read permission)",
             bg=BG, fg="#999", font=FONT_S, anchor="w").pack(fill="x",
                                                               pady=(0, 8))

    # ── Model selector
    r3 = tk.Frame(body, bg=BG)
    r3.pack(fill="x", pady=(0, 8))
    tk.Label(r3, text="Model:", bg=BG, font=FONT).pack(side="left")
    mod_var = tk.StringVar(value=PRIMARY_MODEL)
    ttk.Combobox(r3, textvariable=mod_var, width=42, font=FONT,
                 state="readonly",
                 values=[PRIMARY_MODEL, FALLBACK_MODEL]).pack(side="left",
                                                               padx=6)

    # ── Status
    status_var = tk.StringVar(value="Load a NLP JSON file to begin.")
    tk.Label(body, textvariable=status_var, bg=BG, fg=MUTED,
             font=FONT_S, anchor="w").pack(fill="x", pady=(0, 4))

    # ── Log
    log = scrolledtext.ScrolledText(body, font=MONO, wrap="word",
                                     height=20, relief="flat",
                                     bg="#fdf6ff", fg="#2a0040")
    log.pack(fill="both", expand=True)

    import sys

    class Redirect:
        def __init__(self, w):
            self.w = w
        def write(self, s):
            self.w.insert("end", s)
            self.w.see("end")
            self.w.update()
        def flush(self):
            pass

    def generate():
        path  = path_var.get()
        token = tok_var.get().strip()

        if not os.path.isfile(path):
            messagebox.showwarning("No file", "Please load a NLP JSON first.")
            return
        if not token:
            messagebox.showwarning(
                "No token",
                "Please enter your Hugging Face token.\n\n"
                "Get a free one at:\nhttps://huggingface.co/settings/tokens\n\n"
                "Steps:\n"
                "1. Create free account at huggingface.co\n"
                "2. Go to Settings → Access Tokens\n"
                "3. New token → Read permission → Generate\n"
                "4. Copy the token (starts with hf_...)"
            )
            return

        log.delete("1.0", "end")
        status_var.set("Generating … this may take several minutes.")
        root.update()

        old = sys.stdout
        sys.stdout = Redirect(log)
        try:
            result = run_visual_pipeline(path, token, out_dir)
            n  = result["total_strips"]
            ng = result["generated_panels"]
            np = result["total_panels"]
            status_var.set(
                f"Done — {n} strip(s), {ng}/{np} images generated. "
                f"Saved to {out_dir}"
            )
            if messagebox.askyesno("Done!", f"{ng}/{np} images generated.\n\n"
                                   f"Strips saved to:\n{os.path.abspath(out_dir)}\n\n"
                                   "Open output folder?"):
                import subprocess, platform
                folder = os.path.abspath(out_dir)
                if platform.system() == "Windows":
                    os.startfile(folder)
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            import traceback
            print(f"\nERROR: {exc}\n{traceback.format_exc()}")
            status_var.set(f"Error: {exc}")
        finally:
            sys.stdout = old

    # ── Buttons
    br = tk.Frame(body, bg=BG)
    br.pack(fill="x", pady=(8, 0))

    tk.Button(br, text="Generate Comic Illustrations",
              command=generate,
              bg="#6a0dad", fg="white", font=FONT_B,
              relief="flat", padx=20, pady=8,
              cursor="hand2").pack(side="right")

    tk.Button(br, text="Clear Log",
              command=lambda: log.delete("1.0", "end"),
              bg=BG, fg=MUTED, font=FONT, relief="flat",
              padx=12, pady=8, cursor="hand2").pack(side="right", padx=6)

    root.mainloop()


# ─────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Neuro-Lens Visual — Step 3 (Comic Generation)"
    )
    p.add_argument("--input",      type=str,
                   help="Path to NLP JSON from Step 2")
    p.add_argument("--token",      type=str,
                   default=os.environ.get("HF_TOKEN", ""),
                   help="HF token (or set HF_TOKEN env var)")
    p.add_argument("--model",      type=str,
                   default=PRIMARY_MODEL)
    p.add_argument("--output-dir", type=str,
                   default=COMIC_DIR)
    args = p.parse_args()

    if args.input and os.path.isfile(args.input):
        if not args.token:
            print("ERROR: No HF token. Set HF_TOKEN or use --token.")
            print("Get free token: https://huggingface.co/settings/tokens")
            return
        run_visual_pipeline(args.input, args.token, args.output_dir)
    else:
        run_gui(out_dir=args.output_dir)


if __name__ == "__main__":
    main()
