"""Gradio question-answer interface for the RAG project.

- Phase 2 built this Gradio UI.
- Phase 3 moved the actual question-answer logic into an event-driven
  workflow (see workflow.py). The UI now simply calls that workflow.

The app reuses the existing Pinecone index and does NOT re-index documents.
Environment variables are read from .env by workflow.py:
COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME.
"""

import gradio as gr

# The full RAG logic now lives in the Phase 3 workflow.
from workflow import answer_question


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, label="Your question", placeholder="Ask about the documents..."),
    outputs=gr.Textbox(lines=10, label="Answer"),
    title="RAG with LlamaIndex - Q&A",
    description="Ask a question and get an answer based on the indexed documents.",
)


if __name__ == "__main__":
    demo.launch()
