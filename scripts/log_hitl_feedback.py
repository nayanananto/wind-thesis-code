import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.hitl.feedback_store import FeedbackRecord, FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser("Log human feedback for a semantic state or forecast artifact.")
    parser.add_argument("--artifact_type", type=str, required=True)
    parser.add_argument("--artifact_id", type=str, required=True)
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--reviewer", type=str, default="human")
    args = parser.parse_args()

    store = FeedbackStore()
    record = FeedbackRecord(
        artifact_type=args.artifact_type,
        artifact_id=args.artifact_id,
        reviewer=args.reviewer,
        action=args.action,
        label=args.label,
        note=args.note,
    )
    path = store.append(record)
    print(f"Logged feedback to {path}")


if __name__ == "__main__":
    main()
