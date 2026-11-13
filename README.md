# Multimodal RAG-Based Question Answering System

An AI-powered document assistant that allows users to upload documents, extract text, process content, and ask questions using Retrieval-Augmented Generation (RAG).

## Key Features

- Upload PDF, DOCX, TXT, Markdown, audio and video files.
- Extract text using document processing and OpenAI Whisper.
- Generate overlapping text chunks.
- Create semantic embeddings using Sentence Transformers.
- Store and retrieve document vectors using FAISS.
- Answer questions using Groq LLM.
- Generate document summaries.
- Interactive Streamlit dashboard.
- Light and Dark themes.
- Download and delete uploaded documents.

## Technology Stack

- Python
- Streamlit
- LangChain / RAG concepts
- Sentence Transformers
- FAISS
- Groq API
- OpenAI Whisper
- FFmpeg
- PyPDF
- Python-docx
- NumPy

## Project Workflow

1. Upload documents.
2. Extract text from documents, audio and video.
3. Perform text chunking.
4. Generate embeddings.
5. Store embeddings in FAISS.
6. Retrieve relevant context.
7. Generate answers using Groq LLM.

## Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Run the application:

```bash
streamlit run app.py
```

## Author

Manas Ranjan Aran

## Note

This project is developed for document-based information retrieval, summarization and question answering using a multimodal RAG workflow.