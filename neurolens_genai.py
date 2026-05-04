# neurolens_genai.py

import google.generativeai as genai
from neurolens_rag import NeuroLensRAG


class NeuroLensGenAI:
    def __init__(self, api_key):
        # Configure Gemini API
        genai.configure(api_key=api_key)

        # Use stable model (works everywhere)
        self.model = genai.GenerativeModel("gemini-3-flash-preview")

        # Initialize RAG
        self.rag = NeuroLensRAG()

    # -------------------------------
    # Core LLM call
    # -------------------------------
    def generate(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print("❌ Gemini Error:", str(e))
            return "Error generating response."

    # -------------------------------
    # Combined Pipeline (Single Call)
    # -------------------------------
    def process_text(self, text):
        # Build RAG index
        self.rag.build_index(text)

        # Retrieve relevant context
        context = self.rag.retrieve(text)

        # Single prompt (avoids rate limits)
        prompt = f"""
You are helping a dyslexic student.

Do the following:

1. Simplify the text (very easy English, short sentences)
2. Give 3 bullet points
3. Explain simply with a real-life example

IMPORTANT: Follow this exact format:

SIMPLIFIED:
<text>

BULLETS:
<points>

EXPLANATION:
<text>

Context:
{context}

Text:
{text}
"""

        output = self.generate(prompt)

        return {
            "full_output": output.strip()
        }