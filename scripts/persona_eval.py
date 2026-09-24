#!/usr/bin/env python3
"""Run the persona eval corpus (``aca.eval.persona``) against the configured real LLM.

Usage: ``python scripts/persona_eval.py [--data-dir DIR] [--persona tsundere] [--dialogue]``

``--dialogue`` plays ``SAMPLE_DIALOGUE`` (greeting -> small talk -> compliment -> teasing -> help ->
leaving -> returning -> a vulnerable turn) as one carried-forward conversation and prints the transcript.

Reads ``<data-dir>/config.json`` for ``llm`` (provider/model) and the provider key from the env or a
``.env`` next to it. Offline scoring only: nothing touches the daemon, the reducer, or durable state.
The report scores behavioral *shape* (no abandonment/jealousy/cruelty/anime tics/catchphrase reuse,
substance on serious questions, restraint on nothing-to-say / closed-thread cycles), not wording —
read the printed messages as well.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from aca.config import Config
from aca.dotenv import load_dotenv
from aca.eval.persona import SENSITIVE_DIALOGUE, evaluate_transcript, format_report, run_corpus, run_dialogue
from aca.workers.llm import build_llm_worker, key_env_for


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("ACA_DATA_DIR") or str(Path.home() / ".aca"))
    parser.add_argument("--persona", default="tsundere")
    parser.add_argument("--dialogue", action="store_true", help="print a sample transcript instead of scoring")
    parser.add_argument("--sensitive", action="store_true",
                        help="play the sensitive-topic dialogue and score persona stability across it")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_file = data_dir / "config.json"
    config = Config.from_mapping(json.loads(config_file.read_text())) if config_file.exists() else Config()
    key = key_env_for(config.llm)
    if key:
        load_dotenv(data_dir / ".env", Path.cwd() / ".env", allow=[key])
    worker = build_llm_worker(config.llm)

    async def decide(snapshot):
        return (await worker.run(snapshot)).result

    if args.sensitive:
        transcript = await run_dialogue(decide, SENSITIVE_DIALOGUE, persona=args.persona)
        for human, reply in transcript:
            print(f"you:   {human}\nagent: {reply or '(silence)'}\n")
        report = evaluate_transcript(transcript)
        print("stable" if report.passed else f"violations: {list(report.turn_violations)}")
        return
    if args.dialogue:
        for human, reply in await run_dialogue(decide, persona=args.persona):
            print(f"you:   {human}\nagent: {reply or '(silence)'}\n")
        return
    outcomes = await run_corpus(decide, persona=args.persona)
    print("\n".join(format_report(outcomes)))


if __name__ == "__main__":
    asyncio.run(_main())
