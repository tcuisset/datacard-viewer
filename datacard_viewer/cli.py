"""Command-line launcher for the packaged Streamlit application."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def card_parser():
    parser = argparse.ArgumentParser(description="Browse Combine datacards and shapes.")
    parser.add_argument("--card", default="", help="Datacard path")
    parser.add_argument(
        "--max-files", type=int, default=8,
        help="Maximum open ROOT files per reader (default: 8)",
    )
    parser.add_argument(
        "--max-mb", type=int, default=128,
        help="Template cache limit in MiB per reader; 0 disables caching (default: 128)",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Shape prefix=replacement (repeatable)",
    )
    parser.add_argument("--compare-card", default="", help="Second datacard to compare")
    parser.add_argument(
        "--compare-path-map",
        action="append",
        default=[],
        help="Comparison shape prefix=replacement (repeatable)",
    )
    parser.add_argument(
        "--mass", type=float, default=125, help="Value for $MASS (default: 125)"
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Extra NAME=value substitution (repeatable)",
    )
    return parser


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    card_parser().parse_args(args)
    env = os.environ.copy()
    # Repository-local .streamlit/config.toml is unavailable in a uvx install.
    # Retain those defaults while allowing Streamlit environment overrides.
    for name, value in {
        "STREAMLIT_SERVER_ADDRESS": "127.0.0.1",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_THEME_BASE": "light",
        "STREAMLIT_THEME_PRIMARY_COLOR": "#176b9a",
    }.items():
        env.setdefault(name, value)
    try:
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).with_name("app.py")),
                "--",
                *args,
            ],
            env=env,
        )
    except KeyboardInterrupt:
        return 130
