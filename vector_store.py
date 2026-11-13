
import os
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_DB_FOLDER = "vector_db"

INDEX_PATH = os.path.join(VECTOR_DB_FOLDER, "faiss_index.bin")
METADATA_PATH = os.path.join(VECTOR_DB_FOLDER, "metadata.json")


# ==========================================
# VECTOR STORE CLASS
# ==========================================

class VectorStore:

    def __init__(self):

        print("Loading embedding model...", flush=True)

        self.model = SentenceTransformer(MODEL_NAME)

        print("Embedding model loaded successfully!", flush=True)

        self.index = None
        self.metadata = []

        self.dimension = self.model.get_sentence_embedding_dimension()

        os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)


    # ======================================
    # CREATE VECTOR DATABASE
    # ======================================

    def create_vector_store(self, chunks):

        if not chunks:
            raise ValueError("No chunks provided!")

        print("Generating embeddings...", flush=True)

        texts = []
        metadata = []

        for i, chunk in enumerate(chunks):

            if isinstance(chunk, str):

                texts.append(chunk)

                metadata.append({
                    "chunk_id": i,
                    "text": chunk,
                    "source": "unknown"
                })

            elif isinstance(chunk, dict):

                text = chunk.get("text", "")

                if not text.strip():
                    continue

                texts.append(text)

                metadata.append({
                    "chunk_id": i,
                    "text": text,
                    "source": chunk.get("source", "unknown")
                })

            else:
                raise TypeError(
                    "Each chunk must be a string or dictionary."
                )

        if not texts:
            raise ValueError("No valid text chunks found!")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        print("Creating FAISS index...", flush=True)

        self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(embeddings)

        self.metadata = metadata

        print(f"Stored {self.index.ntotal} vectors successfully!")

        self.save_vector_store()


    # ======================================
    # SAVE VECTOR DATABASE
    # ======================================

    def save_vector_store(self):

        if self.index is None:
            raise ValueError("Vector index has not been created!")

        faiss.write_index(self.index, INDEX_PATH)

        with open(METADATA_PATH, "w", encoding="utf-8") as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=4
            )

        print("Vector database saved successfully!")


    # ======================================
    # LOAD VECTOR DATABASE
    # ======================================

    def load_vector_store(self):

        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("FAISS index not found!")

        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError("Metadata file not found!")

        self.index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as file:

            self.metadata = json.load(file)

        if self.index.ntotal != len(self.metadata):
            raise ValueError("Index and metadata count do not match!")

        print("Vector database loaded successfully!")

        print(f"Total vectors: {self.index.ntotal}")


    # ======================================
    # SEMANTIC SEARCH
    # ======================================

    def search(self, query, top_k=5):

        if self.index is None:
            raise ValueError("Load or create the vector database first!")

        if not query.strip():
            raise ValueError("Search query cannot be empty!")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            result = self.metadata[idx].copy()

            result["similarity_score"] = float(score)

            results.append(result)

        return results


# ==========================================
# TESTING
# ==========================================

if __name__ == "__main__":

    sample_chunks = [
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "source": "AI_Notes.pdf"
        },
        {
            "text": "Deep learning uses neural networks with multiple layers.",
            "source": "AI_Notes.pdf"
        },
        {
            "text": "Python is a popular programming language for data science.",
            "source": "Python_Notes.pdf"
        }
    ]

    # Create vector store
    vector_db = VectorStore()

    vector_db.create_vector_store(sample_chunks)

    # Search test
    query = "What is deep learning?"

    results = vector_db.search(query, top_k=2)

    print("\nSearch Results:")

    for result in results:

        print("\nSource:", result["source"])

        print("Text:", result["text"])

        print("Similarity:", result["similarity_score"])
