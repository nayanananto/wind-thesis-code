from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


def _safe_json_extract(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        pass

    # A provider may wrap a complete object in a Markdown fence or short prose.
    # Decode complete objects rather than treating every JSON value as a dict.
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
            value = part.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts).strip()


def is_plain_text_answer(raw: Any) -> bool:
    """Accept prose, not JSON values or incomplete structured-output envelopes.

    This is a format guard, not a check that the answer is factually grounded.
    Be conservative around JSON/fence syntax so broken output is not presented
    as a successful free-form answer merely because parsing returned ``None``.
    """
    if not isinstance(raw, str) or not raw.strip():
        return False
    text = raw.strip()
    try:
        json.loads(text)
    except (ValueError, TypeError):
        pass
    else:
        # Scalars, strings and lists are JSON too, but are not answer objects.
        return False
    if text.startswith(("[", '"')) or any(marker in text for marker in ("```", "{", "}")):
        return False
    if re.search(r":\s*\[|\bjson\b[\s\S]*\[", text, re.IGNORECASE):
        return False
    return True


def _safe_error_details(exc: Exception) -> dict[str, Any]:
    """Keep only an allowlisted error category and HTTP status, never provider text."""
    status = getattr(exc, "status_code", None)
    if type(status) is not int or not 100 <= status <= 599:
        status = None
    name = type(exc).__name__.lower()
    if status == 429 or "ratelimit" in name:
        category, retryable = "rate_limit", True
    elif status in {408, 504} or "timeout" in name:
        category, retryable = "timeout", True
    elif status == 401 or "authentication" in name:
        category, retryable = "authentication", False
    elif status == 403 or "permission" in name:
        category, retryable = "permission_denied", False
    elif status == 404 or "notfound" in name:
        category, retryable = "model_or_endpoint_not_found", False
    elif status in {400, 405, 422} or "badrequest" in name:
        category, retryable = "invalid_request", False
    elif status is not None and status >= 500:
        category, retryable = "provider_error", True
    elif "connection" in name:
        category, retryable = "connection_error", True
    else:
        category, retryable = "request_error", False
    details: dict[str, Any] = {"category": category, "retryable": retryable}
    if status is not None:
        details["status_code"] = status
    return details


@dataclass
class SemanticReasoner:
    model: str = "gpt-5.5"
    temperature: float = 0.2
    request_timeout: int = 20
    _client: Any = None
    last_error: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self._client = None
            return

        try:
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=os.getenv("SEMANTIC_LLM_MODEL", self.model),
                temperature=self.temperature,
                timeout=self.request_timeout,
            )
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def invoke_json(self, prompt: str) -> dict[str, Any] | None:
        result, _ = self.invoke_json_with_raw(prompt)
        return result

    def invoke_text(self, prompt: str) -> str | None:
        self.last_error = None
        if not self._client:
            self.last_error = {"category": "client_unavailable", "retryable": False}
            return None
        try:
            response = self._client.invoke(prompt)
            content = getattr(response, "content", response)
            text = _response_text(content)
            if not text:
                self.last_error = {"category": "empty_response", "retryable": True}
            return text or None
        except Exception as exc:
            self.last_error = _safe_error_details(exc)
            return None

    def invoke_json_with_raw(self, prompt: str) -> tuple[dict[str, Any] | None, str | None]:
        raw = self.invoke_text(prompt)
        if raw is None:
            return None, None
        result = _safe_json_extract(raw)
        if result is None:
            self.last_error = {"category": "invalid_json", "retryable": False}
        return result, raw
