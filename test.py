import os
from neurolens_genai import NeuroLensGenAI

api_key = os.environ.get("GEMINI_API_KEY")

genai = NeuroLensGenAI(api_key)

text = "Photosynthesis is the process by which plants convert light energy into chemical energy."

result = genai.process_text(text)

print("\n===== OUTPUT =====\n")
print(result["full_output"])