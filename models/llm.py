"""
LLM module for handling xAI (Grok) API interactions.
Provides a clean interface for generating responses.
"""

from openai import OpenAI
from config.config import (
    XAI_API_KEY,
    XAI_BASE_URL,
    LLM_MODEL,
    LLM_FALLBACK_MODELS,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


def _resolve_provider_settings() -> tuple[str, str, str, list[str]]:
    """Resolve provider/base URL/model list from the configured API key and settings."""
    key = (XAI_API_KEY or "").strip()
    base_url = (XAI_BASE_URL or "").strip() or "https://api.x.ai/v1"
    model = (LLM_MODEL or "").strip()
    fallback_models = [m for m in LLM_FALLBACK_MODELS if m]

    # Groq keys start with gsk_. Route to Groq-compatible endpoint/models automatically.
    if key.startswith("gsk_"):
        provider = "groq"
        if "api.x.ai" in base_url:
            base_url = "https://api.groq.com/openai/v1"

        if not model or model.startswith("grok-"):
            model = "llama-3.1-8b-instant"

        if not fallback_models or all(m.startswith("grok-") for m in fallback_models):
            fallback_models = [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
            ]
    else:
        provider = "xai"

    return provider, base_url, model, fallback_models


def _build_model_candidates(primary_model: str, fallback_models: list[str]) -> list[str]:
    """Build an ordered, de-duplicated model candidate list."""
    candidates = [primary_model, *fallback_models]
    deduped: list[str] = []
    for model_name in candidates:
        if model_name and model_name not in deduped:
            deduped.append(model_name)
    return deduped


def get_response(prompt: str, system_prompt: str = None) -> str:
    """
    Generate a response using the xAI Grok API.

    Args:
        prompt: The user's prompt/question
        system_prompt: Optional system prompt for context

    Returns:
        The generated response string, or an error message if failure occurs
    """
    try:
        if not XAI_API_KEY:
            return "Error: xAI API key is not configured. Please set XAI_API_KEY in your .env file."

        provider, resolved_base_url, resolved_model, resolved_fallbacks = _resolve_provider_settings()
        client = OpenAI(api_key=XAI_API_KEY, base_url=resolved_base_url)

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        candidates = _build_model_candidates(resolved_model, resolved_fallbacks)
        last_error = None

        for model_name in candidates:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS
                )
                return response.choices[0].message.content.strip()
            except Exception as model_error:
                err = str(model_error).lower()
                # If this model name is invalid/unavailable, try the next candidate.
                if "model" in err and ("not available" in err or "not found" in err or "does not exist" in err):
                    last_error = model_error
                    continue

                # Any other error should be handled by outer error handling.
                raise

        if last_error is not None:
            raise RuntimeError(
                f"No available model from candidates: {', '.join(candidates)}. Last error: {last_error}"
            )

        raise RuntimeError("No model candidates configured for LLM requests.")

    except Exception as e:
        error_message = str(e)

        if "insufficient_quota" in error_message.lower() or "quota" in error_message.lower():
            return "Error: LLM API quota exceeded. Please check your xAI plan and billing details."
        elif "api_key" in error_message.lower() or "authentication" in error_message.lower():
            return "Error: Invalid API key. Please check your provider key configuration."
        elif "rate_limit" in error_message.lower():
            return "Error: LLM API rate limit exceeded. Please try again later."
        elif "model" in error_message.lower():
            provider, _, resolved_model, resolved_fallbacks = _resolve_provider_settings()
            tried_models = ", ".join(_build_model_candidates(resolved_model, resolved_fallbacks))
            return (
                f"Error: No configured {provider} model is available for this API key/account. "
                f"Tried: {tried_models}."
            )
        else:
            return f"Error generating response: {error_message}"


def get_response_with_context(
    query: str,
    context: str,
    response_mode: str = "concise"
) -> str:
    """
    Generate a response with RAG context.

    Args:
        query: The user's question
        context: Retrieved context from documents or web search
        response_mode: Either "concise" or "detailed"

    Returns:
        The generated response string
    """
    if response_mode.lower() == "detailed":
        mode_instruction = (
            "Provide a detailed, comprehensive explanation with examples where appropriate. "
            "Break down complex concepts and ensure thorough coverage of the topic."
        )
    else:
        mode_instruction = (
            "Answer briefly and clearly. Be concise but informative. "
            "Focus on the key points without unnecessary elaboration."
        )

    system_prompt = f"""You are a helpful AI assistant.
{mode_instruction}

Use the following context to answer the user's question. If the context doesn't contain
relevant information to answer the question, say so clearly and provide what help you can.

Context:
{context}
"""

    return get_response(query, system_prompt)


def get_response_without_context(query: str, response_mode: str = "concise") -> str:
    """
    Generate a response without any context (fallback mode).

    Args:
        query: The user's question
        response_mode: Either "concise" or "detailed"

    Returns:
        The generated response string
    """
    if response_mode.lower() == "detailed":
        mode_instruction = (
            "Provide a detailed, comprehensive explanation with examples where appropriate."
        )
    else:
        mode_instruction = "Answer briefly and clearly. Be concise but informative."

    system_prompt = f"""You are a helpful AI assistant. {mode_instruction}"""

    return get_response(query, system_prompt)
