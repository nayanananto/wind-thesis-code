"""Create a submission ZIP without Git history or local identity strings."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
BLOCKED_MARKERS = ("c:\\users\\", "c:/users/", "raw.githubusercontent.com/")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def anonymity_issues(files: list[Path]) -> list[str]:
    issues: list[str] = []
    for path in files:
        if path.resolve() == Path(__file__).resolve():
            continue
        relative = path.relative_to(ROOT).as_posix()
        lowered_name = relative.lower()
        if any(marker in lowered_name for marker in BLOCKED_MARKERS):
            issues.append(relative)
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig", errors="ignore").lower()
        if any(marker in text for marker in BLOCKED_MARKERS):
            issues.append(relative)
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "wind-thesis-anonymous-artifact.zip",
    )
    args = parser.parse_args()

    files = tracked_files()
    issues = anonymity_issues(files)
    if issues:
        raise SystemExit("Identity/local-path markers remain in: " + ", ".join(issues))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "archive",
            "--format=zip",
            "--prefix=wind-thesis-artifact/",
            f"--output={args.output}",
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"Created {args.output}")
    print("The ZIP contains no .git directory or commit history.")


if __name__ == "__main__":
    main()
