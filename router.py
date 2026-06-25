"""Phase 5: A simple router that picks the right tool for a user request.

Phases 2-4 each do one job:
  - workflow.py   -> free-text Q&A
  - extraction.py -> structured JSON extraction

Phase 5 adds a small "router" on top. It looks at what the user typed and
decides where to send the request:

  - "extraction" -> the structured JSON extraction (extraction.py)
  - "qa"         -> the normal Q&A workflow (workflow.py)

The routing is simple, rule-based keyword matching so it is easy to follow.
No new dependencies and no API keys are hardcoded; the environment variables
(COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME) are still handled by
workflow.py.

Run a quick test from the command line:

    uv run router.py "What is Spirit of Kiro?"
    uv run router.py "Extract the project summary as JSON"
"""

# Phase 3: free-text question answering.
from workflow import answer_question

# Phase 4: structured JSON extraction.
from extraction import extract_project_summary_json


# If the user's request contains any of these words/phrases, we route to the
# structured extraction instead of the normal Q&A workflow.
EXTRACTION_KEYWORDS = [
    "json",
    "structured data",
    "structured",
    "extract",
    "extraction",
    "fields",
    "project summary",
    "technologies list",
    "list of technologies",
    "aws services list",
    "list of aws services",
    "development commands",
]


def classify_request(user_request: str) -> str:
    """Return "extraction" or "qa" for a given user request.

    Simple rule: if the request mentions any extraction keyword, choose
    "extraction"; otherwise choose "qa".
    """
    text = (user_request or "").lower()
    for keyword in EXTRACTION_KEYWORDS:
        if keyword in text:
            return "extraction"
    return "qa"


def answer_with_routing(user_request: str) -> str:
    """Classify the request, run the chosen tool, and return the response.

    The response starts with a small line showing which route was selected.
    """
    if not user_request or not user_request.strip():
        return "Please type a request."

    route = classify_request(user_request)
    print(f"[router] route selected: {route}")

    if route == "extraction":
        result = extract_project_summary_json()
    else:
        result = answer_question(user_request)

    return f"Route selected: {route}\n\n{result}"


if __name__ == "__main__":
    # Quick command-line test:
    #   uv run router.py "What is Spirit of Kiro?"
    #   uv run router.py "Extract the project summary as JSON"
    import sys

    request = " ".join(sys.argv[1:]) or "What is Spirit of Kiro?"
    print("\nRequest:", request)
    print()
    print(answer_with_routing(request))
