import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import re
import os
import time

# =========================
# CONFIGURATION
# =========================
GOOGLE_API_KEY = "AIzaSyD9Jhui2T17yz7AMDk9XrztNPGykmUWKoQ" # Get from aistudio.google.com
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash') # Gemini 1.5 Flash is great for RAG

# =========================
# STEP 1 & 2: DATA LOAD & CLEAN (Keeping your logic, but more robust)
# =========================

def load_data():
    texts = []
    # Load PDFs/Txt
    if os.path.exists("data"):
        for file in os.listdir("data"):
            if file.endswith(".txt"):
                with open(os.path.join("data", file), "r", encoding="utf-8") as f:
                    texts.append(f.read())
    
    # Load CSV
    if os.path.exists("train.csv"):
        df = pd.read_csv("train.csv")
        if "Question" in df.columns and "Answer" in df.columns:
            texts.extend((df["Question"].astype(str) + " " + df["Answer"].astype(str)).tolist())
    return texts

def clean_text(text):
    text = re.sub(r'\s+', ' ', text) # Remove extra whitespace/newlines
    text = re.sub(r'[^a-zA-Z0-9.,()\n\- ]+', ' ', text)
    return text.strip()

# =========================
# STEP 4: ENHANCED CHUNKING (Recursive Style)
# =========================

def chunk_text(text, chunk_size=600, overlap=100):
    chunks = []
    # Simple recursive-style split by length with overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "text": chunk.strip(),
            "type": detect_type(chunk)
        })
        start += (chunk_size - overlap)
    return chunks

def detect_type(text):
    t = text.lower()
    if any(word in t for word in ["condition", "refers to", "defined as"]): return "definition"
    if any(word in t for word in ["symptom", "sign", "pain"]): return "symptoms"
    if any(word in t for word in ["treat", "therapy", "surgery", "medication"]): return "treatment"
    return "general"

# =========================
# STEP 6: BUILD INDEX
# =========================

def build_index(all_texts):
    print("Building Index...")
    all_chunks = []
    for text in all_texts:
        cleaned = clean_text(text)
        all_chunks.extend(chunk_text(cleaned))

    # Deduplicate
    unique_chunks = list({c["text"]: c for c in all_chunks}.values())
    
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [c["text"] for c in unique_chunks]
    embeddings = embed_model.encode(texts, normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "faiss_index.bin")
    np.save("chunks.npy", np.array(unique_chunks, dtype=object))
    print(f"✅ Indexed {len(unique_chunks)} chunks.")

# =========================
# STEP 9: QUERY SYSTEM WITH GEMINI
# =========================

def query_system():
    # Load assets
    index = faiss.read_index("faiss_index.bin")
    chunks = np.load("chunks.npy", allow_pickle=True)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    while True:
        query = input("\nMedical Question (or 'exit'): ")
        if query.lower() == "exit": break

        # 1. Vector Search
        query_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
        _, indices = index.search(query_emb, 15)
        
        retrieved = [chunks[i] for i in indices[0] if i != -1]
        
        # 2. Reranking
        pairs = [[query, c["text"]] for c in retrieved]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(retrieved, scores), key=lambda x: x[1], reverse=True)
        top_chunks = [c["text"] for c, _ in ranked[:4]] # Take top 4

        context = "\n---\n".join(top_chunks)

        # 3. Generation with Gemini
        prompt = f"""You are a professional medical assistant. Use the provided context to answer the question.
        If the answer isn't in the context, say you don't know based on the provided data.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        
        ANSWER:"""

        try:
            response = model.generate_content(prompt)
            print("\n🔍 GEMINI RESPONSE:\n", response.text)
        except Exception as e:
            print(f"Error: {e}. If it's a 429, you're hitting the free tier limit.")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    choice = input("1. Build Index\n2. Query System\nChoice: ")
    if choice == "1":
        data = load_data()
        if data: build_index(data)
        else: print("No data found!")
    elif choice == "2":
        query_system()