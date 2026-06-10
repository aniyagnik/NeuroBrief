"""LLM API clients (Together + Hugging Face)."""

import os

import requests

from neurobrief.config import HF_API_URL, HF_MODEL, LLM_PROVIDER, TOGETHER_API_URL, TOGETHER_MODEL, logger


class LLMError(RuntimeError):
    pass


def clean_env(key):
    return (os.getenv(key) or "").strip().strip('"').strip("'")


def parse_chat_response(result):
    choices = result.get("choices") or []
    if not choices:
        raise LLMError("LLM returned no choices.")
    msg = choices[0].get("message") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        text = (choices[0].get("text") or "").strip()
    if not text:
        raise LLMError("LLM returned an empty response.")
    return text


def together_chat(user_content, max_tokens, temperature, top_p=None):
    api_key = clean_env("TOGETHER_API_KEY")
    if not api_key:
        raise LLMError("TOGETHER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": TOGETHER_MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        body["top_p"] = top_p

    response = requests.post(TOGETHER_API_URL, json=body, headers=headers, timeout=120)
    if not response.ok:
        detail = response.text[:500] if response.text else response.reason
        raise LLMError(f"Together API error ({response.status_code}): {detail}")
    return parse_chat_response(response.json())


def hf_chat(user_content, max_tokens, temperature, top_p=None):
    api_key = clean_env("HF_API_KEY")
    if not api_key:
        raise LLMError("HF_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        body["top_p"] = top_p

    response = requests.post(HF_API_URL, json=body, headers=headers, timeout=120)
    if not response.ok:
        detail = response.text[:500] if response.text else response.reason
        raise LLMError(f"Hugging Face API error ({response.status_code}): {detail}")
    return parse_chat_response(response.json())


def llm_chat(user_content, max_tokens, temperature, top_p=None):
    """Call configured LLM provider(s). auto = Together first, then Hugging Face."""
    provider = LLM_PROVIDER
    errors = []
    logger.info("LLM request provider=%s max_tokens=%d", provider, max_tokens)

    if provider in ("auto", "together"):
        try:
            result = together_chat(user_content, max_tokens, temperature, top_p)
            logger.info("LLM response ok provider=together chars=%d", len(result))
            return result
        except LLMError as e:
            errors.append(str(e))
            logger.warning("Together LLM failed: %s", e)
            if provider == "together":
                raise

    if provider in ("auto", "huggingface", "hf"):
        try:
            result = hf_chat(user_content, max_tokens, temperature, top_p)
            logger.info("LLM response ok provider=huggingface chars=%d", len(result))
            return result
        except LLMError as e:
            errors.append(str(e))
            logger.warning("Hugging Face LLM failed: %s", e)
            raise

    raise LLMError(
        errors[0]
        if errors
        else "No LLM provider configured. Set TOGETHER_API_KEY or HF_API_KEY."
    )
