"""
Web Search module using DuckDuckGo.
Provides fallback search capabilities when local documents don't have answers.
"""

from duckduckgo_search import DDGS
from config.config import WEB_SEARCH_RESULTS


def search_web(query: str, max_results: int = None) -> list[dict]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default from config)

    Returns:
        List of dictionaries containing search results
    """
    try:
        max_results = max_results or WEB_SEARCH_RESULTS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        processed_results = []
        for result in results:
            processed_results.append({
                "title": result.get("title", ""),
                "body": result.get("body", ""),
                "href": result.get("href", ""),
                "source": "web"
            })

        return processed_results

    except Exception as e:
        # Return empty list on error, don't crash the app
        return []


def format_web_results(results: list[dict]) -> str:
    """
    Format web search results into a context string.

    Args:
        results: List of search result dictionaries

    Returns:
        Formatted context string
    """
    if not results:
        return ""

    context_parts = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        body = result.get("body", "No content available")
        href = result.get("href", "")

        context_parts.append(
            f"[Web Source {i}: {title}]\n{body}\nURL: {href}"
        )

    return "\n\n---\n\n".join(context_parts)


def search_web_and_format(query: str, max_results: int = None) -> tuple[str, list[dict]]:
    """
    Search the web and return both formatted context and raw results.

    Args:
        query: Search query string
        max_results: Maximum number of results

    Returns:
        Tuple of (formatted_context, raw_results)
    """
    results = search_web(query, max_results)
    context = format_web_results(results)
    return context, results


def is_web_search_available() -> bool:
    """
    Check if web search is available by making a test query.

    Returns:
        True if web search is working, False otherwise
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text("test", max_results=1))
            return len(results) > 0
    except Exception:
        return False
