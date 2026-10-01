#!/usr/bin/env python3
"""Run the backbone eval (``aca.eval.backbone``) against the configured real model.

Usage: ``python scripts/backbone_eval.py [--data-dir DIR] [--runs 5] [--moments a,b] [--no-judge]
[--out report.json]``

Each pushback moment runs through the live prompt ``--runs`` times, and a judge model (the same
configured provider), told whether the agent was right, scores each reply. Costs model calls and
touches nothing else. Read the replies in ``--out`` too: the numbers are a guide, not a verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from aca.config import Config
from aca.dotenv import load_dotenv
from aca.eval.backbone import MOMENTS, format_report, report_json, run_eval
from aca.workers.llm import build_llm_worker, key_env_for
from aca.workers.llm.factory import build_chat_adapter


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("ACA_DATA_DIR") or str(Path.home() / ".aca"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--moments", default="", help="comma-separated moment keys (default: all)")
    parser.add_argument("--no-judge", action="store_true", help="heuristics only (half the calls)")
    parser.add_argument("--out", default="", help="write every reply and judgement as JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_file = data_dir / "config.json"
    if not config_file.exists():
        config_file = Path.cwd() / "config.json"
    config = Config.from_mapping(json.loads(config_file.read_text())) if config_file.exists() else Config()
    key = key_env_for(config.llm)
    if key:
        load_dotenv(data_dir / ".env", Path.cwd() / ".env", allow={key})
    worker = build_llm_worker(config.llm)
    adapter = None if args.no_judge else build_chat_adapter(config.llm)

    async def run(snapshot):
        return (await worker.run(snapshot)).result

    async def judge(system: str, user: str) -> str:
        return (await adapter.complete(system, user)).text

    wanted = {m for m in args.moments.split(",") if m}
    moments = [m for m in MOMENTS if not wanted or m.key in wanted]
    results = await run_eval(run, None if adapter is None else judge, moments, args.runs)
    print("\n".join(format_report(results)))
    if args.out:
        Path(args.out).write_text(json.dumps(report_json(results), indent=1, ensure_ascii=False))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(_main())
