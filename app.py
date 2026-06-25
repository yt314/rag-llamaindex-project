"""Gradio interface for the RAG project.

- Phase 2 built this Gradio UI.
- Phase 3 moved the question-answer logic into an event-driven workflow
  (see workflow.py). The "Q&A" tab calls that workflow.
- Phase 4 added structured data extraction (see extraction.py). The
  "Structured Extraction" tab returns a JSON summary of the project.
- Phase 5 added a simple router (see router.py). The "Smart Router" tab
  automatically chooses between Q&A and extraction based on the request.

This file only builds the user interface. All RAG logic lives in the other
modules and is reused here unchanged. The app reuses the existing Pinecone
index and does NOT re-index documents. Environment variables are read from
.env by workflow.py: COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME.
"""

import gradio as gr

# Phase 3: free-text question answering via the event-driven workflow.
from workflow import answer_question

# Phase 4: structured JSON extraction.
from extraction import extract_project_summary_json

# Phase 5: automatic routing between Q&A and extraction.
from router import answer_with_routing


# A little light styling to make the app look more polished.
CUSTOM_CSS = """
#app-header { text-align: center; margin-bottom: 4px; }
#app-subtitle { text-align: center; color: #6b7280; margin-top: 0; }

/* My custom footer: a clean centered card, sitting just ABOVE the default footer. */
#app-footer {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 44px;            /* leaves room for the default footer below */
    z-index: 1000;
    width: max-content;
    max-width: 90%;
    margin: 0 auto;
    text-align: center;
    color: #4b5563;
    font-size: 0.95em;
    line-height: 1.5;
    padding: 12px 28px;
    background: var(--background-fill-secondary, #f9fafb);
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
}
#app-footer strong { font-weight: 600; font-size: 1.1em; color: #111827; }
#app-footer a { color: #6b7280; text-decoration: none; }
#app-footer a:hover { text-decoration: underline; }

/* The default Gradio footer: fixed at the very bottom, UNDER my footer. */
footer {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1000;
    padding: 6px 0;
    background: var(--body-background-fill, #ffffff);
}

/* Make room so the two fixed footers never cover the tabs or outputs. */
.gradio-container { padding-bottom: 170px !important; }
"""


with gr.Blocks(title="RAG Assistant for Spirit of Kiro") as demo:
    # ----- Header -----
    gr.Markdown("# RAG Assistant for Spirit of Kiro", elem_id="app-header")
    gr.Markdown(
        "Ask questions, extract structured JSON, or let the smart router choose "
        "the right flow.",
        elem_id="app-subtitle",
    )

    # ----- Tabs (same three flows, same functions) -----
    with gr.Tabs():
        # Tab 1: Phase 2/3 question answering.
        with gr.Tab("Q&A"):
            gr.Markdown("Ask a question and get an answer based on the indexed documents.")
            qa_input = gr.Textbox(
                lines=2, label="Your question", placeholder="Ask about the documents..."
            )
            qa_button = gr.Button("Ask", variant="primary")
            qa_output = gr.Textbox(lines=10, label="Answer")
            # Run on button click or when pressing Enter in the textbox.
            qa_button.click(fn=answer_question, inputs=qa_input, outputs=qa_output)
            qa_input.submit(fn=answer_question, inputs=qa_input, outputs=qa_output)

        # Tab 2: Phase 4 structured extraction (no input needed).
        with gr.Tab("Structured Extraction"):
            gr.Markdown(
                "Extract a structured JSON summary of the project from the indexed documents."
            )
            extraction_button = gr.Button("Extract JSON", variant="primary")
            extraction_output = gr.Code(label="Structured JSON", language="json")
            extraction_button.click(
                fn=extract_project_summary_json, inputs=None, outputs=extraction_output
            )

        # Tab 3: Phase 5 smart router.
        with gr.Tab("Smart Router"):
            gr.Markdown(
                "Type anything. The router sends it to Q&A or to structured "
                "extraction automatically."
            )
            router_input = gr.Textbox(
                lines=2,
                label="Your request",
                placeholder='Try "What is Spirit of Kiro?" or "Extract the project summary as JSON"',
            )
            router_button = gr.Button("Run", variant="primary")
            router_output = gr.Textbox(lines=12, label="Response (with selected route)")
            router_button.click(fn=answer_with_routing, inputs=router_input, outputs=router_output)
            router_input.submit(fn=answer_with_routing, inputs=router_input, outputs=router_output)

    # ----- Footer (fixed at the bottom via CSS) -----
    gr.Markdown(
        "**Created by Yehudit Pollock**  \n"
        "[GitHub Repository](https://github.com/yt314/rag-llamaindex-project) "
        "&nbsp;|&nbsp; "
        "[GitHub Profile](https://github.com/yt314)",
        elem_id="app-footer",
    )


if __name__ == "__main__":
    # In Gradio 6 the theme and custom CSS are passed to launch().
    demo.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS)
