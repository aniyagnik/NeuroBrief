"""Hugging Face LLM client for summary and quiz generation."""

import os

import requests

from neurobrief.config import HF_API_URL, HF_MODEL, logger


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


def llm_chat(user_content, max_tokens, temperature, top_p=None):
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

    logger.info("LLM request provider=huggingface max_tokens=%d", max_tokens)
    response = requests.post(HF_API_URL, json=body, headers=headers, timeout=120)
    if not response.ok:
        detail = response.text[:500] if response.text else response.reason
        raise LLMError(f"Hugging Face API error ({response.status_code}): {detail}")
    result = parse_chat_response(response.json())
    logger.info("LLM response ok provider=huggingface chars=%d", len(result))
    return result
