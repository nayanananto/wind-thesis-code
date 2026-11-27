"""Render the compiled HITL LangGraph using LangGraph's built-in drawing API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.hitl.controlled_agent_graph import ControlledHITLAgentGraph
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod


def main() -> None:
    parser = argparse.ArgumentParser()
    renderer = parser.add_mutually_exclusive_group()
    renderer.add_argument(
        "--png-api",
        action="store_true",
        help="Render through mermaid.ink; this sends the graph topology to that service.",
    )
    renderer.add_argument(
        "--png-local",
        action="store_true",
        help="Render locally with Pyppeteer, which must already be installed.",
    )
    args = parser.parse_args()

    docs_dir = PROJECT_ROOT / "figures"
    docs_dir.mkdir(parents=True, exist_ok=True)
    pipeline_stub = SimpleNamespace(
        feedback_store=SimpleNamespace(path=PROJECT_ROOT / "runtime" / "graph_feedback.jsonl")
    )
    compiled_graph = ControlledHITLAgentGraph(pipeline_stub).graph
    drawable = compiled_graph.get_graph()

    mermaid_path = docs_dir / "langgraph-workflow.mmd"
    png_path = docs_dir / "langgraph-workflow.png"
    mermaid_path.write_text(drawable.draw_mermaid(), encoding="utf-8")

    print(f"Mermaid: {mermaid_path}")
    if args.png_api or args.png_local:
        draw_method = MermaidDrawMethod.API if args.png_api else MermaidDrawMethod.PYPPETEER
        png_bytes = drawable.draw_mermaid_png(
            output_file_path=str(png_path),
            draw_method=draw_method,
            background_color="white",
        )
        if not png_path.exists():
            png_path.write_bytes(png_bytes)
        print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
