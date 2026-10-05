"""Groq chat client: JSON-only replies, schema validation with repair retries,
tokens-per-minute pacing, and a disk cache (saves free-tier quota during development).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .errors import PipelineError

M = TypeVar("M", bound=BaseModel)


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("reply contains no JSON object")
        data = json.loads(text[a : b + 1])
    if not isinstance(data, dict):
        raise ValueError("reply JSON is not an object")
    return data


class LLMClient:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        role: str,
        cache_dir: Path | None = None,
        tpm_limit: int | None = None,
        reasoning_effort: str | None = None,
    ):
        if not api_key:
            raise PipelineError(
                "GROQ_API_KEY is not set. Create a free key at console.groq.com and put it in .env.", stage=role
            )
        try:
            from groq import Groq
        except ImportError as e:  # pragma: no cover
            raise PipelineError("The 'groq' package is not installed (pip install groq).", stage=role) from e
        self.client = Groq(api_key=api_key, max_retries=4, timeout=180)
        self.model = model
        self.role = role
        self.cache_dir = Path(cache_dir) / "llm" if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tpm_limit = tpm_limit
        self._window: deque[tuple[float, int]] = deque()
        self.extra: dict = {}
        if reasoning_effort and "gpt-oss" in model:
            self.extra["reasoning_effort"] = reasoning_effort
        self.calls = 0
        self.cache_hits = 0

    # ---------------------------------------------------------------- public
    def call_json(self, system: str, user: str, schema: type[M], max_tokens: int = 4096, temperature: float = 0.0) -> M:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_err = None
        for _ in range(3):
            raw = self._complete(messages, max_tokens, temperature)
            try:
                return schema.model_validate(_parse_json(raw))
            except (ValueError, ValidationError) as e:
                last_err = e
                messages = messages + [
                    {"role": "assistant", "content": raw[:6000]},
                    {
                        "role": "user",
                        "content": f"That reply was invalid ({str(e)[:600]}). Reply again with ONLY the corrected JSON object.",
                    },
                ]
        raise PipelineError(f"{self.role} ({self.model}) did not return valid JSON after 3 attempts: {last_err}", stage=self.role)

    # --------------------------------------------------------------- private
    def _pace(self, tokens: int) -> None:
        if not self.tpm_limit:
            return
        while True:
            now = time.time()
            while self._window and now - self._window[0][0] > 60:
                self._window.popleft()
            used = sum(t for _, t in self._window)
            if not self._window or used + tokens <= self.tpm_limit:
                break
            time.sleep(max(0.5, 60 - (now - self._window[0][0]) + 0.5))
        self._window.append((time.time(), tokens))

    def _output_cap(self, prompt_est: int) -> int | None:
        """Largest max_tokens that keeps prompt + max_tokens under the tokens/minute limit.
        Groq rejects (413) any single request whose prompt + max_tokens exceeds that limit."""
        if not self.tpm_limit:
            return None
        return self.tpm_limit - prompt_est - 300  # margin for tokenizer differences

    def _complete(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        key = hashlib.sha256(
            json.dumps([self.model, messages, max_tokens, temperature, self.extra], sort_keys=True).encode()
        ).hexdigest()
        cache_file = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache_file and cache_file.exists():
            self.cache_hits += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))["content"]

        import groq

        prompt_est = int(sum(len(m["content"]) for m in messages) / 3.2) + 50
        cap = self._output_cap(prompt_est)
        if cap is not None and cap < 512:
            raise PipelineError(
                f"{self.role}: the prompt alone (~{prompt_est} tokens) nearly fills the {self.tpm_limit} tokens/minute "
                f"limit of '{self.model}' - lower the chunk size in config.py.",
                stage=self.role,
            )
        max_tokens = min(max_tokens, cap) if cap is not None else max_tokens
        self._pace(prompt_est + min(max_tokens, 1500))
        params = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **self.extra,
        )
        content = None
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(**params)
                choice = resp.choices[0]
                content = choice.message.content or ""
                limit = min(16384, cap) if cap is not None else 16384
                if choice.finish_reason == "length" and params["max_tokens"] < limit:
                    params["max_tokens"] = min(limit, params["max_tokens"] * 2)
                    continue
                break
            except groq.BadRequestError as e:
                msg = str(e)
                if "json_validate_failed" in msg and "response_format" in params:
                    params.pop("response_format")  # let the model answer freely, we parse ourselves
                    continue
                if self.extra and any(k in msg for k in self.extra):
                    for k in self.extra:
                        params.pop(k, None)
                    continue
                raise PipelineError(f"{self.role}: Groq rejected the request ({msg[:300]})", stage=self.role) from e
            except groq.RateLimitError as e:
                raise PipelineError(
                    f"{self.role}: Groq free-tier rate limit reached for '{self.model}'. Wait a minute (or until the "
                    f"daily quota resets) and retry, or set a different model in .env. Details: {str(e)[:300]}",
                    stage=self.role,
                ) from e
            except groq.AuthenticationError as e:
                raise PipelineError(f"{self.role}: Groq API key was rejected.", stage=self.role) from e
            except groq.APIConnectionError as e:
                raise PipelineError(f"{self.role}: cannot reach the Groq API ({e}).", stage=self.role) from e
            except groq.APIStatusError as e:
                if e.status_code == 413:
                    # Groq counts prompt + max_tokens against the per-minute limit; shrink the output budget and retry.
                    m = re.search(r"Limit (\d+), Requested (\d+)", str(e))
                    if m and attempt < 3:
                        over = int(m.group(2)) - int(m.group(1))
                        new_max = params["max_tokens"] - over - 200
                        if new_max >= 512:
                            params["max_tokens"] = new_max
                            cap = new_max
                            continue
                    raise PipelineError(
                        f"{self.role}: request too large for '{self.model}' on the free tier - lower the chunk size in config.",
                        stage=self.role,
                    ) from e
                raise PipelineError(f"{self.role}: Groq error {e.status_code}: {str(e)[:300]}", stage=self.role) from e
        if content is None:
            raise PipelineError(f"{self.role}: no reply from Groq.", stage=self.role)
        self.calls += 1
        if cache_file:
            cache_file.write_text(json.dumps({"model": self.model, "content": content}), encoding="utf-8")
        return content
