"""
Helper utilities for the chatbot application.
Contains common utility functions used across modules.
"""

import re
from datetime import datetime


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_sources(sources: list[dict], source_type: str = "document") -> str:
    """
    Format source information for display.

    Args:
        sources: List of source dictionaries
        source_type: Type of sources ("document" or "web")

    Returns:
        Formatted source string for display
    """
    if not sources:
        return ""

    formatted = []

    for i, source in enumerate(sources, 1):
        if source_type == "web":
            title = source.get("title", "Unknown")
            url = source.get("href", "")
            formatted.append(f"{i}. **{title}**\n   {url}")
        else:
            content = truncate_text(source.get("content", ""), 200)
            page = source.get("metadata", {}).get("page", None)
            page_info = f" (Page {page + 1})" if page is not None else ""
            score = source.get("score", 0)
            formatted.append(f"{i}. {page_info} Score: {score:.3f}\n   {content}")

    return "\n\n".join(formatted)


def get_timestamp() -> str:
    """
    Get current timestamp in readable format.

    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sanitize_query(query: str) -> str:
    """
    Sanitize user query by removing extra whitespace and trimming.

    Args:
        query: Raw user query

    Returns:
        Sanitized query string
    """
    if not query:
        return ""

    # Remove extra whitespace and trim
    sanitized = " ".join(query.split())
    return sanitized.strip()


def calculate_context_relevance(results: list[dict]) -> str:
    """
    Calculate and describe the relevance of search results.

    Args:
        results: List of search result dictionaries with scores

    Returns:
        Relevance description string
    """
    if not results:
        return "No results found"

    scores = [r.get("score", float("inf")) for r in results]
    avg_score = sum(scores) / len(scores)

    if avg_score < 0.5:
        return "High relevance"
    elif avg_score < 1.0:
        return "Good relevance"
    elif avg_score < 1.5:
        return "Moderate relevance"
    else:
        return "Low relevance"


def create_chat_message(role: str, content: str, sources: list = None) -> dict:
    """
    Create a standardized chat message dictionary.

    Args:
        role: Message role ("user" or "assistant")
        content: Message content
        sources: Optional list of sources used

    Returns:
        Chat message dictionary
    """
    message = {
        "role": role,
        "content": content,
        "timestamp": get_timestamp()
    }

    if sources:
        message["sources"] = sources

    return message


def generate_local_pdf_answer(query: str, sources: list[dict], response_mode: str = "concise") -> str:
    """
    Create a lightweight extractive answer from retrieved PDF chunks without using an LLM.

    Args:
        query: User question
        sources: Retrieved document chunks
        response_mode: concise or detailed

    Returns:
        Human-readable summary grounded in retrieved chunks
    """
    if not sources:
        return "I could not find relevant content in the PDF for that question."

    query_terms = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2]
    if not query_terms:
        query_terms = ["document"]

    ranked = []
    for item in sources:
        content = item.get("content", "")
        lowered = content.lower()
        term_hits = sum(1 for t in query_terms if t in lowered)
        score = item.get("score", 9999)
        page = item.get("metadata", {}).get("page")
        page_label = f"Page {page + 1}" if page is not None else "Unknown page"
        ranked.append((term_hits, -float(score), page_label, content))

    ranked.sort(reverse=True, key=lambda x: (x[0], x[1]))
    max_items = 5 if response_mode.lower() == "detailed" else 3
    selected = ranked[:max_items]

    lines = ["LLM quota is unavailable, so this answer is generated directly from your PDF excerpts:"]
    for idx, (_, _, page_label, content) in enumerate(selected, 1):
        cleaned = " ".join(content.split())
        snippet = truncate_text(cleaned, 500 if response_mode.lower() == "detailed" else 260)
        lines.append(f"{idx}. ({page_label}) {snippet}")

    return "\n\n".join(lines)
