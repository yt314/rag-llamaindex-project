"""Phase 4: Structured data extraction from the indexed documents.

Phases 2 and 3 answer a question with free text. Phase 4 does something
different: it retrieves relevant chunks from Pinecone and asks the LLM to
return a *structured JSON object* describing the project, using ONLY the
retrieved context.

It reuses the retriever and LLM that were already built in workflow.py
(Phase 3), so it does NOT open a second Pinecone connection and does NOT
re-index any documents. Environment variables (COHERE_API_KEY,
PINECONE_API_KEY, PINECONE_INDEX_NAME) are handled by workflow.py.

Run a quick test from the command line:

    uv run extraction.py
"""

import json
from typing import Any, Dict, List

from llama_index.core.llms import ChatMessage

# Reuse the components already created in Phase 3 (single Pinecone connection).
from workflow import _retriever, _llm


# The fields we want the LLM to fill in, based only on the retrieved context.
EXTRACTION_FIELDS = [
    "project_name",            # string
    "project_type",            # string
    "main_features",           # list of strings
    "core_game_loop",          # string
    "ai_integrations",         # list of strings
    "frontend_technologies",   # list of strings
    "backend_technologies",    # list of strings
    "aws_services",            # list of strings
    "development_commands",    # list of strings
]

# A few targeted queries so we pull chunks covering every field above,
# not just the top 3 for a single query.
EXTRACTION_QUERIES = [
    "project name, type and overview",
    "main features and core game loop",
    "AI integrations and AWS services used",
    "frontend and backend technologies and frameworks",
    "development commands and how to run the project",
]


def gather_context() -> str:
    """Retrieve relevant chunks for all extraction queries and merge them.

    Duplicate chunks (same node id) are kept only once.
    """
    seen: Dict[str, Any] = {}
    for query in EXTRACTION_QUERIES:
        for node_with_score in _retriever.retrieve(query):
            seen[node_with_score.node.node_id] = node_with_score
    print(f"[extraction] gathered {len(seen)} unique chunk(s) from Pinecone")
    return "\n\n".join(node.get_content() for node in seen.values())


def _build_prompt(context_text: str) -> str:
    """Build a prompt that asks the LLM for strict JSON with our fields."""
    return (
        "You are extracting structured information about a software project.\n"
        "Use ONLY the context below. If something is not in the context, use\n"
        'an empty string "" for text fields or an empty list [] for list fields.\n'
        "Do not invent anything.\n\n"
        "Return ONLY a valid JSON object (no markdown, no extra text) with "
        "exactly these keys:\n"
        "- project_name: string\n"
        "- project_type: string (a short phrase describing what kind of project "
        'it is, e.g. the type of game or application such as '
        '"infinite crafting workshop game". Look for a descriptive phrase in the '
        "context and use it; only leave it empty if nothing describes the type)\n"
        "- main_features: list of strings\n"
        "- core_game_loop: string\n"
        "- ai_integrations: list of strings\n"
        "- frontend_technologies: list of strings\n"
        "- backend_technologies: list of strings\n"
        "- aws_services: list of strings\n"
        "- development_commands: list of strings\n\n"
        f"Context:\n{context_text}\n\n"
        "JSON:"
    )


def _parse_json(text: str) -> Dict[str, Any]:
    """Parse the LLM output into a dict, tolerating markdown code fences."""
    cleaned = text.strip()

    # Remove a leading ```json / ``` fence and trailing ``` if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[len("json"):]

    # Keep only the outermost { ... } block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    return json.loads(cleaned)


def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    """Make sure every expected field exists, even if the LLM skipped some."""
    list_fields = {
        "main_features",
        "ai_integrations",
        "frontend_technologies",
        "backend_technologies",
        "aws_services",
        "development_commands",
    }
    result: Dict[str, Any] = {}
    for field_name in EXTRACTION_FIELDS:
        if field_name in list_fields:
            value = data.get(field_name, [])
            result[field_name] = value if isinstance(value, list) else [value]
        else:
            result[field_name] = data.get(field_name, "")
    return result


def extract_project_summary() -> Dict[str, Any]:
    """Retrieve context and return a structured project summary as a dict."""
    context_text = gather_context()
    prompt = _build_prompt(context_text)

    # Cohere uses the Chat API (the old Generate/complete API was removed).
    response = _llm.chat([ChatMessage(role="user", content=prompt)])
    raw_text = str(response.message.content)

    try:
        data = _parse_json(raw_text)
    except (json.JSONDecodeError, ValueError):
        # If the model returned something unparseable, surface it clearly.
        return {"error": "Could not parse JSON from the model.", "raw_output": raw_text}

    print("[extraction] structured JSON extracted")
    return _normalize(data)


def extract_project_summary_json() -> str:
    """Same as extract_project_summary() but returns a pretty JSON string.

    This is convenient for the command line and for the Gradio UI.
    """
    return json.dumps(extract_project_summary(), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Quick command-line test:  uv run extraction.py
    print(extract_project_summary_json())
