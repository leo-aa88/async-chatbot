#!/usr/bin/env python3
"""Run the Monika eval (``aca.eval.monika``) against the configured real LLM.

Usage: ``python scripts/monika_eval.py [--data-dir DIR] [--dialogue]``

``--dialogue`` plays ``SAMPLE_DIALOGUE`` as one carried-forward conversation and prints the transcript.

Reads ``<data-dir>/config.json`` for ``llm`` (provider/model) and the provider key from the env or a
``.env`` next to it. Offline scoring only: nothing touches the daemon, the reducer, or durable state.
The report scores the knowledge boundary, character breaks, and reply length, not wording; read the
printed replies as well.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from aca.config import Config
from aca.dotenv import load_dotenv
from aca.eval.monika import format_report, run_corpus, run_dialogue
from aca.workers.llm import build_llm_worker, key_env_for


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("ACA_DATA_DIR") or str(Path.home() / ".aca-monika"))
    parser.add_argument("--dialogue", action="store_true", help="print a sample transcript instead of scoring")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_file = data_dir / "config.json"
    config = Config.from_mapping(json.loads(config_file.read_text())) if config_file.exists() else Config()
    key = key_env_for(config.llm)
    if key:
        load_dotenv(data_dir / ".env", Path.cwd() / ".env", allow=[key])
    worker = build_llm_worker(config.llm)
    name = config.identity.name or "Monika"

    async def decide(snapshot):
        return (await worker.run(snapshot)).result

    if args.dialogue:
        for human, reply in await run_dialogue(decide, name=name):
            print(f"you:    {human}\nmonika: {reply or '(silence)'}\n")
        return
    print("\n".join(format_report(await run_corpus(decide, name=name))))


if __name__ == "__main__":
    asyncio.run(_main())
