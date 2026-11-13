
import json
from pathlib import Path


# ==========================================
# 1. PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

METADATA_FOLDER = BASE_DIR / "metadata"
CHUNK_FOLDER = BASE_DIR / "chunks"

CHUNK_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CHUNKING CONFIGURATION
# ==========================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ==========================================
# 3. CREATE CHUNKS
# ==========================================

def create_chunks(records, chunk_size=CHUNK_SIZE,
                  chunk_overlap=CHUNK_OVERLAP):

    """
    Split extracted text into overlapping chunks.

    Preserve source metadata such as:
    - source file
    - source type
    - timestamps
    - page numbers
    - line numbers
    - paragraph numbers
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be >= 0 and smaller than chunk size."
        )

    if not records:
        return []

    # Store each record's position in the combined text.
    combined_parts = []
    record_positions = []

    current_position = 0

    for record in records:

        text = record.get("text", "").strip()

        if not text:
            continue

        start_position = current_position
        end_position = start_position + len(text)

        combined_parts.append(text)

        record_positions.append({
            "start": start_position,
            "end": end_position,
            "metadata": record
        })

        current_position = end_position + 1

    combined_text = "\n".join(combined_parts)

    if not combined_text:
        return []

    chunks = []

    start = 0
    chunk_number = 1

    # ======================================
    # 4. SPLIT TEXT INTO CHUNKS
    # ======================================

    while start < len(combined_text):

        end = min(start + chunk_size, len(combined_text))

        chunk_text = combined_text[start:end].strip()

        if not chunk_text:
            break

        # Find original records belonging to this chunk.
        source_references = []

        for position in record_positions:

            # Check whether the record overlaps this chunk.
            if (
                position["start"] < end
                and position["end"] > start
            ):

                metadata = position["metadata"]

                if metadata not in source_references:
                    source_references.append(metadata)

        if source_references:

            source_file = source_references[0].get("source_file")
            source_type = source_references[0].get("source_type")

        else:
            source_file = None
            source_type = None

        # Create structured chunk.
        chunk = {
            "chunk_id": f"chunk_{chunk_number}",
            "text": chunk_text,
            "source_file": source_file,
            "source_type": source_type,
            "source_references": source_references
        }

        chunks.append(chunk)

        chunk_number += 1

        if end == len(combined_text):
            break

        # Move forward while retaining overlap.
        start = end - chunk_overlap

    return chunks


# ==========================================
# 5. PROCESS METADATA FILES
# ==========================================

def process_metadata_files():

    if not METADATA_FOLDER.exists():
        print("Metadata folder does not exist.")
        return

    metadata_files = list(METADATA_FOLDER.glob("*.json"))

    if not metadata_files:
        print("No metadata JSON files found.")
        return

    for metadata_path in metadata_files:

        print(f"\nProcessing: {metadata_path.name}")

        try:

            # Read extracted metadata.
            with open(metadata_path, "r", encoding="utf-8") as file:
                records = json.load(file)

            if not isinstance(records, list):
                raise ValueError(
                    "Expected a list of extracted metadata records."
                )

            # Generate chunks.
            chunks = create_chunks(records)

            # Save chunks.
            output_path = CHUNK_FOLDER / (
                metadata_path.stem + "_chunks.json"
            )

            with open(output_path, "w", encoding="utf-8") as file:

                json.dump(
                    chunks,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(f"Total chunks created: {len(chunks)}")
            print(f"Saved: {output_path.name}")

        except Exception as error:

            print(
                f"Error processing {metadata_path.name}: {error}"
            )


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    process_metadata_files()

    print("\nChunking completed successfully!")
