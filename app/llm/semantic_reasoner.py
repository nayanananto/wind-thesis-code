from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


def _safe_json_extract(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


@dataclass
class SemanticReasoner:
    model: str = "gpt-5.5"
    temperature: float = 0.2
    request_timeout: int = 20
    _client: Any = None

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
        if not self._client:
            return None
        try:
            response = self._client.invoke(prompt)
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = "".join(str(part) for part in content)
            return _safe_json_extract(str(content))
        except Exception:
            return None

    def invoke_text(self, prompt: str) -> str | None:
        if not self._client:
            return None
        try:
            response = self._client.invoke(prompt)
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = "".join(str(part) for part in content)
            text = str(content).strip()
            return text or None
        except Exception:
            return None

    def invoke_json_with_raw(self, prompt: str) -> tuple[dict[str, Any] | None, str | None]:
        raw = self.invoke_text(prompt)
        if raw is None:
            return None, None
        return _safe_json_extract(raw), raw
