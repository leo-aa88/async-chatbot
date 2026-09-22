# Voice input (`aca chat --voice`)

Speak instead of (or as well as) typing. Voice is a **client-side perception adapter**: the CLI
captures the microphone, a lightweight VAD carves the stream into **acoustic utterances**, a Whisper
model transcribes each, a **turn assembler** joins them into a conversational turn once you yield the
floor, and only that committed turn is injected through the **exact same durable ingress as typed
input** — a `HumanMessage` tagged `input_mode="voice"`. To cognition, a spoken turn and a typed turn
are both human utterances; **nothing in the reducer or cognition branches on `input_mode`**, so
determinism and replay are untouched (invariant 2).

```
you speak → microphone → VAD (speech boundaries) → Whisper (small.en) → utterance text
         → turn assembler (floor-yield) → one HumanMessage (same ingress as typing)
         → Wolfy thinks → (maybe) speaks
```

Voice is entirely optional and lives **alongside** typing — plain `aca chat` needs none of the
voice dependencies, and `--voice` adds the microphone without taking typing away.

## Install & run

```bash
pip install -e ".[voice]"     # faster-whisper + sounddevice + numpy
```

Then select a real transcriber in `<data-dir>/config.json` (the default provider is the offline
`fake` test double, which `--voice` refuses — see below):

```json
{ "voice": { "provider": "faster-whisper", "model": "small.en" } }
```

```bash
aca service start             # in one terminal
aca chat --voice              # in another: type OR speak; Ctrl-D to quit
```

The first run downloads the model weights. `device` and `compute_type` default to `auto`: with a
CUDA GPU present, `small.en` loads at `int8_float16` (~2 GB VRAM, fits a 4 GB card); with no CUDA it
falls back to CPU at `int8` — it works, just slower. (An explicit CUDA-only `compute_type` like
`int8_float16` set against a CPU box is rejected up front, before the download, rather than crashing
mid-load.)

Without the extra installed, `aca chat` still works for typing; `--voice` fails fast with a clear
"pip install 'aca[voice]'" message rather than erroring deep in the capture loop.

**Why `fake` is refused on the live path.** `provider: "fake"` is a deterministic test double that
emits placeholder text (`utterance of N samples`), not a transcription. Wiring it to a running
agent would commit fabricated human turns to durable ingress on any noise the VAD trips, so
`aca chat --voice` refuses it with a hint to set `voice.provider=faster-whisper`. The fake stays a
test-only double.

## Model choice

The default `small.en` at `int8_float16` needs ~2 GB VRAM (fits a 4 GB RTX 3050) and is
English-only. For lower latency at a small accuracy cost, try `distil-small.en`. Both load through
the same code path — only `voice.model` changes.

| model               | role                      |
| ------------------- | ------------------------- |
| `small.en`          | default; better accuracy  |
| `distil-small.en`   | prioritize responsiveness |

## Configuration (`config.json`, `"voice"` section)

All defaults are offline and dependency-free (`provider: "fake"`, `vad: "energy"`), so tests and
dry runs never touch a model or a microphone.

| key                | default          | meaning                                                        |
| ------------------ | ---------------- | -------------------------------------------------------------- |
| `provider`         | `fake`           | `fake` (offline) or `faster-whisper`                           |
| `model`            | `small.en`       | Whisper model id                                               |
| `device`           | `auto`           | `cuda` / `cpu` / `auto` (auto → CUDA if present, else CPU)      |
| `compute_type`     | `auto`           | `auto` picks per device (`int8_float16` CUDA / `int8` CPU)      |
| `language`         | `en`             | transcription language                                         |
| `vad`              | `energy`         | `energy` (RMS gate, no deps) or `silero` (needs `voice-silero`)|
| `vad_threshold`    | `0.02`           | energy-VAD speech threshold, normalized RMS in `[0,1]`         |
| `silence`          | `0.6s`           | end-of-utterance silence (hangover) that closes one *utterance* |
| `min_utterance`    | `0.3s`           | shorter runs of speech are dropped as blips                    |
| `max_utterance`    | `30s`            | force-cut so a stuck mic can't buffer without bound            |
| `turn_gap`         | `1.3s`           | trailing silence that commits an ordinary *turn* (must be > `silence`) |
| `continuation_gap` | `2.8s`           | longer grace when the turn looks syntactically unfinished      |
| `max_turn`         | `120s`           | safety cap; forces a turn commit at the next silence boundary  |

**Tuning turn-taking.** `silence` is when *one utterance* ends; `turn_gap` is when *your whole turn*
ends. Raise `turn_gap` if Wolfy answers before you've finished a thought; lower it if replies feel
sluggish. `continuation_gap` only applies when your last word looks unfinished (a trailing "and",
"to", "uh", …), so a mid-thought pause waits longer than a finished sentence.

## Design notes

- **Utterances vs turns (DESIGN §36).** The [segmenter](../src/aca/voice/segmenter.py) answers
  *"where did speech stop?"* — it yields an **acoustic utterance** (speech bounded by silence). The
  [turn assembler](../src/aca/voice/turn_assembler.py) answers *"has the human yielded the floor?"* —
  it joins utterances across thinking pauses and commits one `HumanMessage` per turn. A VAD boundary
  is not a turn boundary, so a mid-thought pause never fires a premature turn. We never stream
  partial transcripts as messages.
- **Floor-yield is measured in acoustic time.** The commit waits on trailing silence counted from
  the *frame stream*, not wall-clock after ASR — so Whisper latency is never mistaken for you pausing.
- **The pure core is deterministic.** `audio`, `vad`, and `segmenter` are frame-based, clock-free,
  and dependency-free, so the boundary logic is exhaustively unit-tested with no audio backend.
  Real capture (`sounddevice`) and transcription (`faster-whisper`) are lazy optional imports.
- **Heavy work off the loop.** A Whisper forward pass runs in a worker thread (`asyncio.to_thread`)
  so it never blocks the client's event loop — the same discipline the daemon follows.
- **Pairs with TTS.** Transcription is the input half; Kokoro-82M TTS (built separately) is the
  output half. Together: speak aloud, and Wolfy answers from the speakers.
