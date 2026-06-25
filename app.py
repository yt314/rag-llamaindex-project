"""Phase 2: Simple Gradio question-answer interface for the RAG project.

This app reuses the Pinecone index that was already created by `prepare.py`.
It does NOT re-index the documents. On every question it only:
  1. Embeds the question with Cohere (input_type="search_query").
  2. Retrieves the most relevant chunks from the existing Pinecone index.
  3. Asks a Cohere LLM to write an answer based on those chunks.
"""

import os

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.vector_stores.pinecone import PineconeVectorStore

# This must match the namespace used in prepare.py when the documents were saved.
PINECONE_NAMESPACE = "kiro-steering"


def get_env_var(name: str) -> str:
    """Read a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Please add it to your .env file."
        )
    return value


def build_query_engine():
    """Connect to the existing Pinecone index and build a query engine once."""
    cohere_api_key = get_env_var("COHERE_API_KEY")
    pinecone_api_key = get_env_var("PINECONE_API_KEY")
    pinecone_index_name = get_env_var("PINECONE_INDEX_NAME")

    # Embeddings for the *question* use input_type="search_query"
    # (prepare.py used "search_document" for the indexed documents).
    embed_model = CohereEmbedding(
        api_key=cohere_api_key,
        model_name="embed-english-v3.0",
        input_type="search_query",
    )

    # Cohere LLM that writes the answer from the retrieved chunks.
    llm = Cohere(api_key=cohere_api_key, model="command-a-03-2025")

    # Connect to the existing Pinecone index (no re-indexing here).
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(pinecone_index_name)

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=PINECONE_NAMESPACE,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    return index.as_query_engine(llm=llm, similarity_top_k=3)


# Build the query engine a single time when the app starts.
query_engine = build_query_engine()


def answer_question(question: str) -> str:
    """Answer a single question using the indexed documents."""
    if not question or not question.strip():
        return "Please type a question."
    try:
        response = query_engine.query(question)
        return str(response)
    except Exception as error:  # keep it simple for a beginner project
        return f"Something went wrong while answering: {error}"


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, label="Your question", placeholder="Ask about the documents..."),
    outputs=gr.Textbox(lines=10, label="Answer"),
    title="RAG with LlamaIndex - Q&A",
    description="Ask a question and get an answer based on the indexed documents.",
)


if __name__ == "__main__":
    demo.launch()
