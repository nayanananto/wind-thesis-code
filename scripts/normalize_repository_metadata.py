from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PORTABLE_ROOTS = ("app", "artifacts", "data", "figures", "results", "scripts", "theory_experiments")


def _portable_string(value: str) -> str:
    normalized = value.replace("\\", "/")
    lower = normalized.lower()
    for directory in PORTABLE_ROOTS:
        marker = f"/{directory.lower()}/"
        position = lower.find(marker)
        if position >= 0:
            return normalized[position + 1 :]
    return value


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        return _portable_string(value)
    return value


def normalize_json_file(path: Path, *, write: bool) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    normalized = _normalize(payload)
    changed = normalized != payload
    if changed and write:
        path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize repository JSON paths for portable clones.")
    parser.add_argument("--write", action="store_true", help="Rewrite files instead of reporting them.")
    args = parser.parse_args()

    changed: list[str] = []
    for path in sorted(ROOT.rglob("*.json")):
        if ".git" in path.parts:
            continue
        try:
            if normalize_json_file(path, write=args.write):
                changed.append(path.relative_to(ROOT).as_posix())
        except json.JSONDecodeError:
            continue

    action = "Updated" if args.write else "Would update"
    print(f"{action} {len(changed)} JSON file(s).")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
