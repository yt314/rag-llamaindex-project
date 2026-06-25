"""Gradio interface for the RAG project.

- Phase 2 built this Gradio UI.
- Phase 3 moved the question-answer logic into an event-driven workflow
  (see workflow.py). The "Q&A" tab calls that workflow.
- Phase 4 added structured data extraction (see extraction.py). The
  "Structured Extraction" tab returns a JSON summary of the project.

The app reuses the existing Pinecone index and does NOT re-index documents.
Environment variables are read from .env by workflow.py:
COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME.
"""

import gradio as gr

# Phase 3: free-text question answering via the event-driven workflow.
from workflow import answer_question

# Phase 4: structured JSON extraction.
from extraction import extract_project_summary_json


# Tab 1: the existing Phase 2/3 question-answer interface (unchanged behavior).
qa_interface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, label="Your question", placeholder="Ask about the documents..."),
    outputs=gr.Textbox(lines=10, label="Answer"),
    title="RAG with LlamaIndex - Q&A",
    description="Ask a question and get an answer based on the indexed documents.",
)

# Tab 2: Phase 4 structured extraction. No input needed - just press submit.
extraction_interface = gr.Interface(
    fn=extract_project_summary_json,
    inputs=[],
    outputs=gr.Code(label="Structured JSON", language="json"),
    title="Structured Extraction",
    description="Extract a structured JSON summary of the project from the indexed documents.",
)

demo = gr.TabbedInterface(
    [qa_interface, extraction_interface],
    ["Q&A", "Structured Extraction"],
    title="RAG with LlamaIndex",
)


if __name__ == "__main__":
    demo.launch()
