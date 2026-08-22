"""OpenAI-compatible API adapter using urllib (no external dependencies)."""

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Optional

from harness.providers.base import ModelProvider, Completion


class OpenAICompatAdapter(ModelProvider):
    """
    OpenAI-compatible API adapter.

    Works with OpenAI API and any compatible endpoint (Azure, local models, etc.).
    Uses only stdlib (urllib) - no external HTTP library required.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4",
        max_retries: int = 3,
        timeout: int = 60,
        pricing: Optional[dict[str, float]] = None
    ):
        """
        Initialize OpenAI-compatible adapter.

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (defaults to OPENAI_BASE_URL env var or OpenAI default)
            model: Model identifier
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            pricing: Pricing dict with 'prompt_per_1k_tokens' and 'completion_per_1k_tokens' keys
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")

        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.pricing = pricing or {
            "prompt_per_1k_tokens": 0.03,
            "completion_per_1k_tokens": 0.06
        }

        # Ensure base_url doesn't end with slash
        self.base_url = self.base_url.rstrip("/")

    def complete(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any]
    ) -> Completion:
        """
        Generate a completion using the API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            params: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Completion object
        """
        start_time = time.time()

        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            **params
        }

        # Remove custom params that aren't part of OpenAI API
        for key in ["item_id", "condition_id"]:
            payload.pop(key, None)

        # Make API request with retries
        response_data = self._make_request_with_retries(payload)

        latency_ms = (time.time() - start_time) * 1000

        # Extract completion data
        choice = response_data["choices"][0]
        message = choice["message"]
        text = message["content"]

        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Check if cached (some providers return this)
        cached = response_data.get("cached", False)

        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
            model_version=response_data.get("model", self.model),
            latency_ms=latency_ms,
            cached=cached,
            metadata={
                "finish_reason": choice.get("finish_reason"),
                "response_id": response_data.get("id")
            }
        )

    def _make_request_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make API request with exponential backoff retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self._make_request(payload)
            except urllib.error.HTTPError as e:
                status_code = e.code
                last_error = e

                # Retry on rate limit (429) or server errors (5xx)
                if status_code == 429 or status_code >= 500:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s, ...
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue

                # Don't retry on client errors (4xx except 429)
                raise

            except urllib.error.URLError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise

        # All retries exhausted
        raise Exception(f"Request failed after {self.max_retries} retries: {last_error}")

    def _make_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make a single API request."""
        url = f"{self.base_url}/chat/completions"

        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST"
        )

        # Make request
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                return response_data
        except urllib.error.HTTPError as e:
            # Try to read error response
            try:
                error_data = json.loads(e.read().decode("utf-8"))
                error_msg = error_data.get("error", {}).get("message", str(e))
            except:
                error_msg = str(e)

            raise Exception(f"API request failed (HTTP {e.code}): {error_msg}")
        except urllib.error.URLError as e:
            raise Exception(f"API request failed (network error): {e.reason}")
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

    def provider_name(self) -> str:
        """Return provider name."""
        return "openai_compat"

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost in USD based on token counts and pricing.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        prompt_cost = (prompt_tokens / 1000) * self.pricing["prompt_per_1k_tokens"]
        completion_cost = (completion_tokens / 1000) * self.pricing["completion_per_1k_tokens"]

        return prompt_cost + completion_cost
