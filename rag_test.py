from neurolens_rag import NeuroLensRAG

rag = NeuroLensRAG()

text = """Photosynthesis is how plants make food. 
Plants use sunlight, water, and carbon dioxide. 
This process produces oxygen."""

rag.build_index(text)

query = "How do plants make food?"

context = rag.retrieve(query)

print("\nRetrieved Context:\n", context)