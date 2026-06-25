"""Phase 3: A simple event-driven RAG workflow built on LlamaIndex Workflows.

Instead of doing retrieval + answering in one function (like Phase 2), the
question-answer process is split into clear *steps*. Each step receives an
*event*, does one small job, updates a shared *state* object, and then sends
the next event.

The flow of events is:

    StartEvent          -> receive_question
    RetrieveEvent       -> retrieve  (get chunks from Pinecone)
    CheckContextEvent   -> check_context  (was anything relevant found?)
        - if nothing found -> StopEvent  (return a "no answer" message)
        - if found         -> GenerateEvent
    GenerateEvent       -> generate  (ask the LLM to write the answer)
    StopEvent           -> final answer returned to the caller

No API keys are hardcoded. Environment variables are read from `.env`:
COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME.
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, List

from dotenv import load_dotenv

load_dotenv()

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

# Must match the namespace used in prepare.py when the documents were saved.
PINECONE_NAMESPACE = "kiro-steering"
SIMILARITY_TOP_K = 3


# --------------------------------------------------------------------------- #
# Environment handling (no hardcoded keys, clear error if something is missing)
# --------------------------------------------------------------------------- #
def get_env_var(name: str) -> str:
    """Read a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Please add it to your .env file."
        )
    return value


# --------------------------------------------------------------------------- #
# Shared workflow state (a simple object passed from step to step)
# --------------------------------------------------------------------------- #
@dataclass
class RAGState:
    """Holds everything we collect while answering one question."""
    question: str = ""
    chunks: List[Any] = field(default_factory=list)  # retrieved nodes
    context_found: bool = False
    answer: str = ""


# --------------------------------------------------------------------------- #
# Events (each one moves the workflow from one step to the next)
# --------------------------------------------------------------------------- #
class RetrieveEvent(Event):
    """Question received, time to retrieve chunks from Pinecone."""
    state: Any


class CheckContextEvent(Event):
    """Chunks retrieved, time to check if anything relevant was found."""
    state: Any


class GenerateEvent(Event):
    """Relevant context found, time to ask the LLM for an answer."""
    state: Any


# --------------------------------------------------------------------------- #
# The workflow: one method per step
# --------------------------------------------------------------------------- #
class RAGWorkflow(Workflow):
    """A small, readable RAG pipeline expressed as workflow steps."""

    def __init__(self, retriever, llm, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.llm = llm

    @step
    async def receive_question(self, ev: StartEvent) -> RetrieveEvent:
        """Step 1: take the user's question and create a fresh state."""
        state = RAGState(question=ev.question)
        print(f"[workflow] received question: {state.question!r}")
        return RetrieveEvent(state=state)

    @step
    async def retrieve(self, ev: RetrieveEvent) -> CheckContextEvent:
        """Step 2: retrieve the most relevant chunks from Pinecone."""
        state = ev.state
        state.chunks = self.retriever.retrieve(state.question)
        print(f"[workflow] retrieved {len(state.chunks)} chunk(s) from Pinecone")
        return CheckContextEvent(state=state)

    @step
    async def check_context(self, ev: CheckContextEvent) -> GenerateEvent | StopEvent:
        """Step 3: decide whether we found anything useful to answer with."""
        state = ev.state
        state.context_found = len(state.chunks) > 0
        if not state.context_found:
            print("[workflow] no relevant context found -> stopping early")
            return StopEvent(
                result="I couldn't find relevant information in the documents "
                "to answer that question."
            )
        print("[workflow] relevant context found -> generating an answer")
        return GenerateEvent(state=state)

    @step
    async def generate(self, ev: GenerateEvent) -> StopEvent:
        """Step 4: ask the LLM to write an answer from the retrieved chunks."""
        state = ev.state
        context_text = "\n\n".join(node.get_content() for node in state.chunks)
        prompt = (
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {state.question}\n\n"
            "Answer:"
        )
        # Cohere uses the Chat API (the old Generate/complete API was removed).
        response = self.llm.chat([ChatMessage(role="user", content=prompt)])
        state.answer = str(response.message.content).strip()
        print("[workflow] answer generated")
        # Step 5: return the final answer.
        return StopEvent(result=state.answer)


# --------------------------------------------------------------------------- #
# Build the retriever + LLM once, then reuse them for every question
# --------------------------------------------------------------------------- #
def build_components():
    """Connect to the existing Pinecone index and create the retriever + LLM."""
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

    retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
    return retriever, llm


# Build everything a single time when this module is imported.
_retriever, _llm = build_components()
_workflow = RAGWorkflow(retriever=_retriever, llm=_llm, timeout=120, verbose=False)


def answer_question(question: str) -> str:
    """Run the event-driven workflow for one question and return the answer.

    This is a normal (synchronous) function so it is easy to call from Gradio.
    """
    if not question or not question.strip():
        return "Please type a question."

    async def _run() -> str:
        # workflow.run() must be awaited from inside a running event loop.
        return await _workflow.run(question=question)

    try:
        return asyncio.run(_run())
    except Exception as error:  # keep it simple for a beginner project
        return f"Something went wrong while answering: {error}"


if __name__ == "__main__":
    # Quick command-line test:  uv run workflow.py "your question"
    import sys

    test_question = " ".join(sys.argv[1:]) or "What are the kiro steering documents about?"
    print("\nQuestion:", test_question)
    print("\nAnswer:", answer_question(test_question))
