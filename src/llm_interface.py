"""
llm_interface.py
----------------
Modular interface for querying LLMs via the OpenRouter API.
Supports easy model swapping and provides a consistent integration layer.
"""

import os
import time
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported OpenRouter model aliases
# ---------------------------------------------------------------------------
OPENROUTER_MODELS = {
    "gpt4o-mini":        "openai/gpt-4o-mini",
    "gpt4o":             "openai/gpt-4o",
    "claude-haiku":      "anthropic/claude-haiku-4-5",
    "claude-sonnet":     "anthropic/claude-sonnet-4-5",
    "llama3-8b":         "meta-llama/llama-3-8b-instruct",
    "llama3-70b":        "meta-llama/llama-3-70b-instruct",
    "mistral-7b":        "mistralai/mistral-7b-instruct",
    "gemini-flash":      "google/gemini-flash-2.5",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMInterface:
    """
    Wrapper around the OpenRouter API.

    Parameters
    ----------
    model : str
        Short alias (see OPENROUTER_MODELS) or a full OpenRouter model string.
    api_key : str or None
        OpenRouter API key.  Falls back to the ``OPENROUTER_API_KEY``
        environment variable when not supplied explicitly.
    temperature : float
        Sampling temperature (default 0.7).
    max_tokens : int
        Maximum tokens in the completion (default 1024).
    max_retries : int
        Number of retry attempts on transient HTTP errors (default 3).
    retry_delay : float
        Seconds to wait between retries (default 2.0).
    site_url : str
        Optional referrer URL sent in the ``HTTP-Referer`` header.
    site_name : str
        Optional site name sent in the ``X-Title`` header.
    """

    def __init__(
        self,
        model: str = "gpt4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        site_url: str = "https://research.example.com",
        site_name: str = "Prompt Evolution Research",
    ):
        self.model = OPENROUTER_MODELS.get(model, model)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set the OPENROUTER_API_KEY environment variable or pass api_key explicitly."
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Send a single-turn query to the LLM.

        Parameters
        ----------
        prompt : str
            User-turn content.
        system_message : str or None
            Optional system message prepended to the conversation.

        Returns
        -------
        str
            The assistant's reply text.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    OPENROUTER_BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else None
                logger.warning("HTTP %s on attempt %d/%d: %s", status, attempt, self.max_retries, e)
                if status == 429 or (status and status >= 500):
                    time.sleep(self.retry_delay * attempt)
                else:
                    raise
            except requests.exceptions.RequestException as e:
                logger.warning("Request error on attempt %d/%d: %s", attempt, self.max_retries, e)
                time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"LLM query failed after {self.max_retries} attempts.")

    def query_with_history(self, messages: list[dict]) -> str:
        """
        Send a multi-turn conversation to the LLM.

        Parameters
        ----------
        messages : list of dict
            OpenAI-style message list, each with ``role`` and ``content`` keys.

        Returns
        -------
        str
            The assistant's reply text.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    OPENROUTER_BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else None
                logger.warning("HTTP %s on attempt %d/%d: %s", status, attempt, self.max_retries, e)
                if status == 429 or (status and status >= 500):
                    time.sleep(self.retry_delay * attempt)
                else:
                    raise
            except requests.exceptions.RequestException as e:
                logger.warning("Request error on attempt %d/%d: %s", attempt, self.max_retries, e)
                time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"LLM query failed after {self.max_retries} attempts.")
