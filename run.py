#!/usr/bin/env python3
"""Local launcher for Wiki Hami Extraction V1.

Examples:
    python run.py --api
    python run.py --api --reload
    python run.py --web
    python run.py --worker
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    root = str(ROOT)
    env["PYTHONPATH"] = root if not current else f"{root}{os.pathsep}{current}"
    return env


def _exec(args: list[str]) -> None:
    os.chdir(ROOT)
    os.execvpe(sys.executable, [sys.executable, *args], _environment())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wiki Hami Extraction V1 locally.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--api", action="store_true", help="Run the FastAPI service.")
    mode.add_argument("--web", action="store_true", help="Run the Streamlit inspector.")
    mode.add_argument("--worker", action="store_true", help="Run the Celery extraction worker.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.web and args.reload:
        parser.error("--reload is only valid with --api")
    if args.worker and (args.reload or args.port is not None):
        parser.error("--worker does not accept --reload or --port")

    if args.api:
        port = args.port or 8000
        command = ["-m", "uvicorn", "app.main:app", "--host", args.host, "--port", str(port)]
        if args.reload:
            command.append("--reload")
        _exec(command)

    if args.worker:
        _exec([
            "-m", "celery",
            "-A", "app.jobs.celery_app:celery_app",
            "worker",
            "--loglevel", os.getenv("WIKI_HAMI_LOG_LEVEL", "INFO").lower(),
            "--concurrency", "1",
        ])

    port = args.port or 8501
    _exec([
        "-m", "streamlit", "run", str(ROOT / "ui" / "streamlit_app.py"),
        "--server.address", args.host,
        "--server.port", str(port),
    ])


if __name__ == "__main__":
    main()
