# Converts 
# video --FFmpeg-> audio -- Whisper--> text
# audio -- Whisper--> text
# text --keep as is -> text

import os
import subprocess
import whisper
Input_Folder="input_files"
Audio_Folder="audio_files"
Output_Folder="text_files"

os.makedirs(Output_Folder, exist_ok=True)
os.makedirs(Audio_Folder, exist_ok=True)

#Load Whisper model
# model=whisper.load_model("base")
model = None

def get_whisper_model():

    global model

    if model is None:
        print("Loading Whisper model...")
        model = whisper.load_model("base")

    return model

def video_to_audio(Video_Path,Audio_path):
    command=["ffmpeg",
            "-i",Video_Path,
            "-vn",
            "-acodec","mp3",
            "-y",
            Audio_path]
    subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)



def audio_to_text(Audio_path):

    result = get_whisper_model().transcribe(
        Audio_path,
        fp16=False
    )
    records = []
    for segment in result["segments"]:

        records.append({
            "start_time": segment["start"],
            "end_time": segment["end"],
            "text": segment["text"].strip()
        })

    return records

# def audio_to_text(Audio_path,Output_path):
#     result=model.transcribe(Audio_path,fp16=False)
#     return result["text"]


from pypdf import PdfReader
from docx import Document

def document_to_text(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    records = []

    # TXT and Markdown
    if extension in [".txt", ".md"]:

        with open(file_path, "r", encoding="utf-8") as f:

            for line_number, line in enumerate(f, start=1):

                if line.strip():

                    records.append({
                        "line_number": line_number,
                        "text": line.strip()
                    })

    # PDF
    elif extension == ".pdf":

        reader = PdfReader(file_path)

        for page_number, page in enumerate(reader.pages, start=1):

            page_text = page.extract_text() or ""

            for line_number, line in enumerate(
                page_text.splitlines(), start=1
            ):

                if line.strip():

                    records.append({
                        "page_number": page_number,
                        "line_number": line_number,
                        "text": line.strip()
                    })

    # Word
    elif extension == ".docx":

        doc = Document(file_path)

        for paragraph_number, paragraph in enumerate(
            doc.paragraphs, start=1
        ):

            if paragraph.text.strip():

                records.append({
                    "paragraph_number": paragraph_number,
                    "text": paragraph.text.strip()
                })

    else:
        raise ValueError("Unsupported document format")

    return records

def create_chunks(records, filename, chunk_size=1000, overlap=150):

    chunks = []

    # Preserve useful location information
    full_text = []

    for record in records:

        text = record["text"]

        if "page_number" in record:
            text = f"[Page {record['page_number']}] {text}"

        elif "start_time" in record:
            text = (
                f"[Time {record['start_time']:.2f}s - "
                f"{record['end_time']:.2f}s] {text}"
            )

        full_text.append(text)

    text = "\n".join(full_text)

    start = 0
    chunk_id = 0

    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end].strip()

        if chunk_text:

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": filename
            })

            chunk_id += 1

        if end >= len(text):
            break

        start = end - overlap

    return chunks

import json

def save_output(name, records, source_type, filename):

    # Add source information to every record
    for record in records:

        record["source_file"] = filename
        record["source_type"] = source_type

    # Save readable text
    text_path = os.path.join(
        Output_Folder,
        name + ".txt"
    )

    with open(text_path, "w", encoding="utf-8") as f:

        for record in records:
            f.write(record["text"] + "\n")

    # Save metadata separately
    metadata_folder = "metadata"
    os.makedirs(metadata_folder, exist_ok=True)

    metadata_path = os.path.join(
        metadata_folder,
        name + ".json"
    )

    with open(metadata_path, "w", encoding="utf-8") as f:

        json.dump(
            records,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Successfully processed: {filename}")


####

def process_input_folder():

    all_chunks = []

    os.makedirs(Input_Folder, exist_ok=True)

    for filename in os.listdir(Input_Folder):

        file_path = os.path.join(Input_Folder, filename)

        name, extension = os.path.splitext(filename)
        extension = extension.lower()

        try:

            if extension in [".mp4", ".mkv", ".avi", ".mov"]:

                audio_path = os.path.join(
                    Audio_Folder,
                    name + ".mp3"
                )

                video_to_audio(file_path, audio_path)

                records = audio_to_text(audio_path)

                save_output(name, records, "video", filename)

            elif extension in [".mp3", ".wav", ".m4a", ".flac"]:

                records = audio_to_text(file_path)

                save_output(name, records, "audio", filename)

            elif extension in [".txt", ".md", ".pdf", ".docx"]:

                records = document_to_text(file_path)

                save_output(
                    name,
                    records,
                    extension.replace(".", ""),
                    filename
                )

            else:
                print(f"Unsupported file format: {filename}")
                continue

            # Convert extracted records into chunks
            chunks = create_chunks(records, filename)

            all_chunks.extend(chunks)

            print(f"Created {len(chunks)} chunks for {filename}")

        except Exception as error:

            print(f"Error processing {filename}: {error}")

    return all_chunks


if __name__ == "__main__":

    chunks = process_input_folder()

    print(f"\nTotal chunks created: {len(chunks)}")




######
