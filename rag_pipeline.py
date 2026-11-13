
import os
import time
from dotenv import load_dotenv
from groq import Groq
from vector_store import VectorStore

load_dotenv()


class RAGPipeline:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please add it to your .env file."
            )

        self.client = Groq(api_key=api_key)

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b"
        )

        self.vector_store = VectorStore()
        self.vector_store.load_vector_store()

        print("RAG Pipeline initialized successfully!")

    def retrieve_context(self, question):

        print("Retrieving context...")

        results = self.vector_store.search(question, top_k=5)

        # Return structured results, not a formatted string
        if not results:
            return []

        if isinstance(results, dict):
            results = results.get("results", [])

        if not isinstance(results, list):
            raise TypeError(
                "VectorStore.search() must return a list of dictionaries."
            )

        return results

    def generate_answer(self, question, context):

        context_text = ""

        for i, item in enumerate(context, start=1):

            if not isinstance(item, dict):
                continue

            chunk_text = item.get("text", "")
            source = item.get("source", "Unknown")
            score = item.get("similarity_score", 0)

            context_text += (
                f"\nContext {i}:\n"
                f"Source: {source}\n"
                f"Content: {chunk_text}\n"
                f"Similarity Score: {score}\n"
            )

        prompt = f"""
You are an intelligent AI Teaching Assistant.

Answer the user's question using ONLY the provided context.

Instructions:
1. Provide accurate and relevant answers.
2. Identify the source when asked where information comes from.
3. Mention the original filename.
4. Never invent document names or information.
5. If the source is unavailable, clearly state that.
6. Mention all relevant sources when multiple documents
   contain the requested information.
7. If the answer is not available in the context,
   say that you could not find sufficient information.

Context:
{context_text}

User Question:
{question}

Answer:
"""

        print("Generating answer using Groq...")

        for attempt in range(3):

            try:

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )

                return response.choices[0].message.content

            except Exception as error:

                if attempt == 2:
                    raise error

                print(
                    f"Generation attempt {attempt + 1} failed: {error}"
                )

                time.sleep(2 ** attempt)

    def ask_question(self, question):

        if not question or not question.strip():
            return "Please enter a valid question."

        try:

            # Step 1: Retrieve relevant chunks
            context = self.retrieve_context(question)

            if not context:
                return (
                    "No relevant information was found "
                    "in the uploaded documents."
                )

            # Step 2: Generate answer
            answer = self.generate_answer(
                question,
                context
            )

            return answer

        except Exception as error:

            return f"Error while processing your question: {error}"


if __name__ == "__main__":

    rag = RAGPipeline()

    print("\nRAG-Based Question Answering System")
    print("-----------------------------------")

    while True:

        question = input(
            "\nAsk a question (type exit to stop): "
        )

        if question.lower().strip() == "exit":
            print("Thank you for using the AI Teaching Assistant!")
            break

        print("\nProcessing your question...")

        answer = rag.ask_question(question)

        print("\nAnswer:")
        print(answer)
