# RAG with LlamaIndex

A simple Retrieval-Augmented Generation (RAG) project built with LlamaIndex,
Cohere (embeddings + LLM) and Pinecone (vector database).

## Phases

- **Phase 1 — Indexing (`prepare.py`):** loads documents from the
  `kiro-steering` folder, splits them into chunks, creates Cohere embeddings,
  and saves them to a Pinecone index. Run this **once** to build the index.
- **Phase 2 — Q&A app (`app.py`):** a small Gradio web app where you type a
  question and get an answer based on the already-indexed documents. It reuses
  the existing Pinecone index and does **not** re-index on every question.

## Setup

1. Install the dependencies (either option works):

   ```bash
   uv sync
   ```

   or, with plain pip:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your keys:

   ```env
   COHERE_API_KEY=your_cohere_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=kiro
   ```

## Step 1 — Build the index (run once)

```bash
uv run prepare.py
```

This reads the documents from `kiro-steering/` and uploads their embeddings to
Pinecone.

## Step 2 — Run the Gradio Q&A app

```bash
uv run app.py
```

Gradio prints a local URL (usually http://127.0.0.1:7860). Open it in your
browser, type a question, and read the answer.

### Example question

> What is the purpose of the kiro steering documents?

The app embeds your question with Cohere, retrieves the most relevant chunks
from Pinecone, and uses the Cohere LLM to write an answer based on them.

## Environment variables

The app reads these from `.env` and shows a clear error if any is missing:

- `COHERE_API_KEY` — Cohere API key (used for embeddings and the LLM)
- `PINECONE_API_KEY` — Pinecone API key
- `PINECONE_INDEX_NAME` — name of the Pinecone index (e.g. `kiro`)
