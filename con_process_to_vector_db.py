
from process import process_input_folder
from vector_store import VectorStore


def build_database():

    print("\nStarting document processing...")

    # Step 1: Extract text and create chunks
    chunks = process_input_folder()

    if not chunks:
        print("No valid document chunks found!")
        return

    print(f"\nTotal chunks: {len(chunks)}")

    # Step 2: Initialize vector store
    vector_db = VectorStore()

    # Step 3: Generate embeddings and save FAISS database
    vector_db.create_vector_store(chunks)

    print("\nVector database created successfully!")


if __name__ == "__main__":

    build_database()
