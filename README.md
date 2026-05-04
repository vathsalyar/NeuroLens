# NeuroLens 🧠
**An LLM and RAG-Based Multimodal Generative AI Framework for Dyslexia-Friendly Learning**

## What it does
Converts standard textbook pages into accessible, multimodal educational comic strips for dyslexic learners.

## Pipeline
1. **OCR** (`neurolens_ocr.py`) — Extracts text from textbook images using PaddleOCR
2. **RAG** (`neurolens_rag.py`) — Chunks, embeds, and retrieves relevant context using all-MiniLM-L6-v2
3. **NLP/LLM** (`neurolens_nlp.py`) — Simplifies text into dyslexia-friendly bullets using Gemini 3 Flash
4. **Visual** (`neurolens_visual.py`) — Generates comic panels using FLUX.1-schnell / Stable Diffusion XL

## Setup

### Step 1 — OCR
```bash
pip install -r requirements_ocr.txt
```

### Step 2 — NLP
```bash
pip install -r requirements_nlp.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

### Step 3 — Visual Generation
```bash
pip install -r requirements_visual.txt
export HF_TOKEN=your_huggingface_token_here
```

## Run
```bash
python neurolens_ocr.py       # Step 1: Extract text
python neurolens_nlp.py       # Step 2: Simplify text
python neurolens_visual.py    # Step 3: Generate comic panels
```