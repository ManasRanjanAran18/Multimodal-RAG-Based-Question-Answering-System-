
print("STEP 1: Script started", flush=True)

import os
print("STEP 2: OS imported", flush=True)

import warnings
warnings.filterwarnings("ignore")
print("STEP 3: Warnings configured", flush=True)

import json
print("STEP 4: JSON imported", flush=True)

import numpy as np
print("STEP 5: NumPy imported", flush=True)

from pathlib import Path
print("STEP 6: Path imported", flush=True)

print("STEP 7: Importing SentenceTransformer...", flush=True)

from sentence_transformers import SentenceTransformer

print("STEP 8: SentenceTransformer imported successfully", flush=True)

print("STEP 9: All imports completed", flush=True)



# ==========================================
# 1. PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CHUNK_FOLDER = BASE_DIR / "chunks"
EMBEDDING_FOLDER = BASE_DIR / "embeddings"

EMBEDDING_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

print("Current file:", __file__, flush=True)
print("Chunks folder:", CHUNK_FOLDER, flush=True)
print("Loading embedding model...", flush=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully!", flush=True)

# ==========================================
# 3. GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(texts):

    """
    Convert a list of text chunks into numerical vectors.
    """

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


# ==========================================
# 4. PROCESS CHUNK FILES
# ==========================================

def process_chunk_files():

    if not CHUNK_FOLDER.exists():
        print("Chunks folder does not exist.")
        return

    chunk_files = list(CHUNK_FOLDER.glob("*_chunks.json"))
    print("Chunk files found:", len(chunk_files))

    for file in chunk_files:
        print("Found:", file.name)

    if not chunk_files:
        print("No chunk JSON files found.")
        return

    for chunk_path in chunk_files:

        print(f"\nProcessing: {chunk_path.name}")

        try:

            # Read chunks
            with open(chunk_path, "r", encoding="utf-8") as file:
                chunks = json.load(file)

            if not isinstance(chunks, list):
                raise ValueError("Expected a list of chunks.")

            if not chunks:
                print("No chunks available.")
                continue

            # Extract text from chunks
            texts = [
                chunk["text"]
                for chunk in chunks
                if chunk.get("text", "").strip()
            ]

            if not texts:
                print("No valid text found.")
                continue

            # Generate embeddings
            embeddings = generate_embeddings(texts)

            # Save numerical vectors
            embedding_path = EMBEDDING_FOLDER / (
                chunk_path.stem + "_vectors.npy"
            )

            np.save(embedding_path, embeddings)

            # Preserve chunk metadata in the same order
            metadata_path = EMBEDDING_FOLDER / (
                chunk_path.stem + "_metadata.json"
            )

            with open(metadata_path, "w", encoding="utf-8") as file:

                json.dump(
                    chunks,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(f"Total embeddings: {len(embeddings)}")
            print(f"Vector dimensions: {embeddings.shape[1]}")
            print(f"Saved vectors: {embedding_path.name}")
            print(f"Saved metadata: {metadata_path.name}")

        except Exception as error:

            print(
                f"Error processing {chunk_path.name}: {error}"
            )


# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    process_chunk_files()

    print("\nEmbedding generation completed!")
