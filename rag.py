import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# 1. Load embedding model
# -----------------------------
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 2. Load documents
# -----------------------------
data_folder = "data"
docs = []
doc_names = []

for file in os.listdir(data_folder):
    if file.endswith(".txt"):
        with open(os.path.join(data_folder, file), "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(text)
            doc_names.append(file)

if len(docs) == 0:
    print("❌ No .txt files found in data folder!")
    exit()

print(f"Loaded {len(docs)} documents:", doc_names)

# -----------------------------
# 3. Create embeddings
# -----------------------------
print("Creating embeddings...")
embeddings = embedder.encode(docs)

# -----------------------------
# 4. Create FAISS index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("✅ Vector database ready!")

# -----------------------------
# 5. Load text generation model
# -----------------------------
print("Loading text generation model (first time may take a while)...")
generator = pipeline("text-generation", model="distilgpt2")

# -----------------------------
# 6. Question loop
# -----------------------------
while True:
    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Bye 👋")
        break

    # Embed the query
    query_vec = embedder.encode([query])

    # Search in FAISS
    k = min(2, len(docs))  # retrieve up to 2 docs (you have 2 files)
    distances, indices = index.search(np.array(query_vec), k)

    print("\n🔎 Retrieved from:")
    retrieved_text = ""
    for idx in indices[0]:
        print(" -", doc_names[idx])
        retrieved_text += docs[idx] + "\n"

    # -----------------------------
    # 7. Limit context size (IMPORTANT to avoid crashes)
    # -----------------------------
    MAX_CONTEXT_CHARS = 1500  # safe limit for small models
    short_context = retrieved_text[:MAX_CONTEXT_CHARS]

    # Build prompt
    prompt = f"""
Use the following context to answer the question.

Context:
{short_context}

Question: {query}
Answer:
"""

    # -----------------------------
    # 8. Generate answer
    # -----------------------------
    result = generator(
        prompt,
        max_new_tokens=200,   # only generate new tokens
        do_sample=True,
        temperature=0.7,
        
    )

    print("\n🧠 Answer:")
    print(result[0]["generated_text"])
