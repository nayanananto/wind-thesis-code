from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

from app.config.paths import HITL_DATA_DIR, ensure_project_dirs


@dataclass
class FeedbackRecord:
    feedback_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    artifact_type: str = ""
    artifact_id: str = ""
    reviewer: str = "human"
    action: str = "note"
    label: str = ""
    note: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class FeedbackStore:
    def __init__(self, path: str | Path | None = None):
        ensure_project_dirs()
        self.path = Path(path or HITL_DATA_DIR / "feedback_log.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: FeedbackRecord) -> Path:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return self.path

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    def filter(
        self,
        artifact_type: str | None = None,
        artifact_id: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.read_all()
        if artifact_type is not None:
            rows = [row for row in rows if row.get("artifact_type") == artifact_type]
        if artifact_id is not None:
            rows = [row for row in rows if row.get("artifact_id") == artifact_id]
        return rows

