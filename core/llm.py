"""OpenAI-compatible chat client that records token usage per call."""

import os
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class Sampling:
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_s: float
    reasoning: str | None = None
    raw: dict = field(default_factory=dict)


class Client:
    """One endpoint, one served model name. Retries connection errors and 5xx with backoff."""

    def __init__(
        self, base_url: str, model: str, api_key_env: str | None = None, timeout: float = 600, retries: int = 6
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retries = retries
        headers = {}
        if api_key_env:
            headers["Authorization"] = f"Bearer {os.environ[api_key_env]}"
        self._http = httpx.Client(timeout=timeout, headers=headers)

    def chat(self, messages: list[dict], seed: int, sampling: Sampling, extra: dict | None = None) -> Reply:
        body = {
            "model": self.model,
            "messages": messages,
            "seed": seed,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            **(extra or {}),
        }
        for attempt in range(self.retries):
            t = time.time()
            try:
                r = self._http.post(f"{self.base_url}/chat/completions", json=body)
                if r.status_code >= 500 or r.status_code == 429:
                    raise httpx.HTTPStatusError(r.text[:200], request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == self.retries - 1:
                    raise
                time.sleep(min(60, 2**attempt))
                last = e  # noqa: F841 - kept for a debugger
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return Reply(
            text=msg.get("content") or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cached_tokens=int(details.get("cached_tokens") or 0),
            latency_s=time.time() - t,
            reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
            raw={"finish_reason": data["choices"][0].get("finish_reason")},
        )
