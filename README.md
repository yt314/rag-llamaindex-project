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
- **Phase 3 — Event-driven workflow (`workflow.py`):** the question-answer
  logic is split into clear, ordered steps using LlamaIndex Workflows. The
  Gradio app now calls this workflow instead of one big function.
- **Phase 4 — Structured data extraction (`extraction.py`):** instead of a
  free-text answer, this retrieves relevant chunks and asks the LLM to return
  a structured **JSON summary** of the project. It is exposed as a second tab
  in the Gradio app.

## Phase 3 — How the workflow works

The work of answering a question is broken into small **steps**. Each step
receives an **event**, does one job, updates a shared **state** object
(`RAGState`), and sends the next event:

```
StartEvent        -> receive_question   (store the question in the state)
RetrieveEvent     -> retrieve           (get the top chunks from Pinecone)
CheckContextEvent -> check_context      (was anything relevant found?)
      |-- nothing found --> StopEvent    (return a polite "no answer" message)
      |-- found         --> GenerateEvent
GenerateEvent     -> generate           (ask the Cohere LLM to write the answer)
StopEvent         -> final answer returned to the app
```

- **State:** a simple `RAGState` dataclass holding `question`, `chunks`,
  `context_found`, and `answer`. It is passed from step to step.
- **Events:** small classes (`RetrieveEvent`, `CheckContextEvent`,
  `GenerateEvent`) plus the built-in `StartEvent` / `StopEvent`.
- **Steps:** the methods of `RAGWorkflow`, one per stage above.

Each step prints a short `[workflow] ...` line so you can watch the events
flow in the terminal. The workflow still reuses the existing Pinecone index
and never re-indexes documents.

## Phase 4 — How the structured extraction works

`extraction.py` turns the documents into a structured JSON object instead of a
free-text answer:

1. It runs a few targeted queries against Pinecone and merges the unique
   chunks (so the context covers every field, not just one topic).
2. It asks the Cohere LLM to return **only** a JSON object, filling in the
   fields below using **only** the retrieved context (empty string / empty
   list when something is not found - it does not invent data).
3. It parses and normalizes the JSON so every field always exists.

The extracted fields are:

```
project_name, project_type, main_features, core_game_loop,
ai_integrations, frontend_technologies, backend_technologies,
aws_services, development_commands
```

It reuses the retriever and LLM already built in Phase 3, so it does **not**
open a second Pinecone connection and never re-indexes documents.

### Run / test the extraction from the command line

```bash
uv run extraction.py
```

This prints the structured JSON to the terminal.

### Run the extraction in the app

`uv run app.py` now opens **two tabs**:

- **Q&A** - the Phase 2/3 question-answer interface.
- **Structured Extraction** - press submit to get the JSON summary.

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

### Testing the workflow directly (without the UI)

You can run the Phase 3 workflow straight from the command line and watch the
event steps print as it runs:

```bash
uv run workflow.py "What is the Spirit of Kiro project and its main features?"
```

## Environment variables

The app reads these from `.env` and shows a clear error if any is missing:

- `COHERE_API_KEY` — Cohere API key (used for embeddings and the LLM)
- `PINECONE_API_KEY` — Pinecone API key
- `PINECONE_INDEX_NAME` — name of the Pinecone index (e.g. `kiro`)
