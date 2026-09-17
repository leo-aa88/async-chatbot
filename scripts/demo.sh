#!/usr/bin/env bash
# Launch a throwaway ACA agent tuned so proactive (unsolicited) messages are easy to observe,
# then drop into an interactive chat. Send a substantive line, then STOP TYPING for ~15s and the
# agent will message you on its own. Everything is torn down on exit.
#
# The aggressive timings here are for demonstration only; production defaults are far calmer
# (ACTIVE window 3m, spontaneous rate 0.25/hr). See docs/DESIGN.md §7.4, §10, §31.1.
set -euo pipefail

ACA="${ACA:-aca}"
DIR="$(mktemp -d "${TMPDIR:-/tmp}/aca-demo.XXXXXX")"

cat > "$DIR/config.json" <<'JSON'
{
  "rng_seed": 7,
  "cognition": {
    "spontaneous_activation_rate_per_hour": 300,
    "semantic_worthiness_floor": 0.3,
    "selection_temperature": 0.4,
    "null_candidate_score": -2.0
  },
  "conversation": { "active_within": "15s", "idle_within": "30m" },
  "timing": {
    "proactive_cooldown": "10s",
    "proactive_ttl": "1h",
    "proactive_burst_window": "5s"
  },
  "budgets": {
    "proactive_messages_per_hour": 100,
    "proactive_llm_calls_per_hour": 200,
    "proactive_llm_calls_per_day": 2000
  }
}
JSON

cleanup() {
  "$ACA" --data-dir "$DIR" service stop >/dev/null 2>&1 || true
  wait 2>/dev/null || true
  rm -rf "$DIR"
}
trap cleanup EXIT INT TERM

"$ACA" --data-dir "$DIR" service start >"$DIR/service.log" 2>&1 &

# Wait for the IPC socket to appear before connecting.
for _ in $(seq 1 100); do
  [ -S "$DIR/aca.sock" ] && break
  sleep 0.1
done
if [ ! -S "$DIR/aca.sock" ]; then
  echo "demo: agent service failed to start; log follows:" >&2
  cat "$DIR/service.log" >&2
  exit 1
fi

cat <<'MSG'
──────────────────────────────────────────────────────────────────────────────
 ACA demo agent (throwaway, deleted on exit).
 Try: say something substantive, then STOP TYPING for ~15 seconds.
 The agent will send a message on its own (no prompt, no reconnect).
 Replies are canned (deterministic fake worker) — this shows the *behavior*.
 Ctrl-D to quit.
──────────────────────────────────────────────────────────────────────────────
MSG

"$ACA" --data-dir "$DIR" chat
