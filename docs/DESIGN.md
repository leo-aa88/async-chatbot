# DESIGN.md — Asynchronous Conversational Agent

**Status:** v0.8 — LLM-proposed discourse relation (§35), extending v0.7 discourse continuity (§34) on the frozen v0.6 core  
**Scope:** Software-first prototype, designed so the same cognitive architecture can later be embodied  
**Primary goal:** Build a persistent conversational agent that emulates human-like introspection and temporal continuity through stochastic attention, memory activation, self-monitoring, inhibition, delayed response, silence, autonomous initiative, and durable identity across runtime interruptions.

> **Architecture status:** The v0.6 core is implementation-frozen. §34 (discourse focus and topic continuity) is the first sanctioned v0.7 extension, admitted under the frozen-core exception restated in §31 on evidence from the running prototype: live traces showed the agent voicing thoughts that were individually worthwhile but conversationally orphaned — semantically continuous, but not *discourse*-continuous. §34 adds one new expression-stage gate and a discourse-focus addition to conversation state. It changes no v0.6 *rule*; it does insert cross-reference pointers into §7.4, §11.1, and §16.3 so a reader following those sections finds the extension. Further structural changes still require evidence from the running prototype or a concrete embodiment requirement.

---

## 1. Summary

Traditional chat systems are request/response machines:

```text
human input -> model -> response
```

This project deliberately breaks that contract.

The agent should behave more like an introspective human conversational partner while preserving a coherent sense of continuity across time:

- it can receive a message and decide not to answer;
- it can receive a message, think about it, and answer later;
- it can initiate a conversation without a human prompt;
- it can revisit unresolved topics after time has passed;
- it can notice patterns in its own recent behavior;
- it can remain silent even when it has an internally active topic;
- it can forget, deprioritize, or suppress thoughts;
- it can change internal state without producing language;
- it should not require a continuously running LLM.

The central design principle is:

> **Human input is optional. Agent output is optional.**

A conversational turn is therefore not defined as one human message followed by one AI response. It is defined as a bounded cognition cycle that may or may not have been triggered by a human and may or may not produce outward language.

---

## 2. Design goals

### 2.1 Primary goals

The system should:

1. **Emulate introspection behaviorally** without making claims about subjective consciousness.
2. **Support autonomous initiative** without fixed cron-like message intervals.
3. **Support meaningful silence** as a first-class action.
4. **Support delayed responses** when a topic remains salient and resurfaces later.
5. **Maintain persistent conversational state and identity** across long periods and process restarts.
6. **Model its own recent behavior** well enough to regulate repetition, interruption, and conversational pressure.
7. **Use stochastic processes** so the same state does not always produce the same conversational behavior.
8. **Keep most autonomous cognition cheap** by running activation, decay, retrieval, and gating outside the generative LLM.
9. **Bound all cognition cycles** so internal reflection cannot recursively trigger uncontrolled reflection.
10. **Remain inspectable**: internal state should be structured and debuggable rather than hidden in a permanent free-form inner monologue.
11. **Preserve temporal continuity** across clean shutdown, host sleep, power loss, and process crash without pretending cognition occurred while the agent was unavailable.
12. **Separate logical agent identity from process/UI lifetime** so closing a client does not terminate the agent and restarting the service does not create a new agent.
13. **Remain portable to embodiment**: the same event/reducer/memory architecture should be able to run later on a Raspberry Pi or similar host while delegating hard real-time control to an MCU.

### 2.2 Secondary goals

The architecture should make it possible later to add:

- external event sources such as GitHub, files, calendar, system state, or sensors;
- multimodal perception;
- local models for low-cost classification and perception;
- multiple personalities or temperament profiles;
- speech input/output and physical embodiment;
- multiple cooperating agents.

## 3. Non-goals

The first version will **not** attempt to:

- claim or test machine consciousness;
- maintain a literal continuous linguistic stream of consciousness;
- continuously invoke a generative LLM while idle;
- simulate biological cognition faithfully;
- model emotions as if they were genuine subjective states;
- build the physical robot itself;
- perform hard real-time motor control;
- provide a general autonomous-agent framework;
- use distributed infrastructure unless the prototype requires it;
- guarantee a response to every low-obligation social message;
- simulate missed thoughts during periods when the agent service was not running.

This project is a behavioral architecture experiment, not a theory of mind. Temporal continuity means persistent identity, memory, and awareness that real time elapsed; it does not imply continuous subjective experience.

## 4. Terminology

### 4.1 Event

Anything that may alter committed agent state.

Examples include a human message, stochastic wake, external event, lifecycle event, scheduled constraint, feedback event, or asynchronous worker result.

### 4.2 Activation

A scalar representing how likely a topic, provisional memory, or unresolved thread is to enter attention. Activation is structured state, not language.

### 4.3 Reflection

Semantic reasoning about an event, topic, memory, or unresolved issue.

Example:

> The user mentioned returning to fluid mechanics. This may be more than nostalgia.

### 4.4 Introspection / metacognition

Reasoning about the agent's own structured state or recent behavior.

Example:

> I have initiated several conversations recently; another interruption is probably unnecessary.

The term **introspection** is operational: the system emulates behavioral patterns associated with introspective humans.

### 4.5 Initiative

The tendency for a valid internal thought to become unsolicited outward communication.

### 4.6 Inhibition

The tendency to suppress or delay an otherwise plausible optional/proactive outward response.

### 4.7 Cognition cycle

One bounded processing pass triggered by a source event or stochastic activation. A cycle may end with speech, deferred speech, state update only, or silence.

### 4.8 Logical agent

The persistent identity represented by durable state: memories, history, temperament, lifecycle history, and an `agent_id`.

The logical agent is **not** identical to one operating-system process.

### 4.9 Agent service / daemon

The long-running local process that hosts the reducer, scheduler, persistence layer, workers, and delivery adapters. On systems where the term is appropriate it may run as a daemon/service.

Stopping this process suspends cognition; it does not create a new logical agent on the next launch.

### 4.10 Client

A human-facing interface such as a CLI, TUI, web UI, desktop UI, or later a speech interface. Clients communicate with the agent service over local IPC/API boundaries.

Closing a client does **not** stop the logical agent.

### 4.11 Runtime session

One contiguous execution of the agent service, identified by a `runtime_session_id`. A single logical agent may have many runtime sessions over its lifetime.

### 4.12 Suspension and resume

A period during which the logical agent remains durably identifiable but the service performs no cognition. Clean shutdown, host power-off, and host sleep are modeled as suspension from the emulation perspective. A crash or power loss is an **unclean interruption** operationally, but on recovery the missing interval is still treated as a period without cognition rather than as a new identity.

### 4.13 Wall time and active time

- **Wall time:** real-world elapsed time whether the service is running or not.
- **Active time:** time during which the agent service is running and capable of processing cognition cycles.

Time-sensitive state explicitly declares which clock semantics it uses.

## 5. Core behavioral contract

The system must support both of these statements:

```text
Human input is optional.
Agent output is optional.
```

This implies four valid interaction cases:

| Human input | Agent output | Meaning |
|---|---:|---|
| yes | yes | ordinary conversational response |
| yes | no | the agent perceives the message but remains silent |
| no | yes | proactive / autonomous initiative |
| no | no | internal state evolution only |

The fourth case is important. The system may have a cognition cycle that produces no language at all.

---

## 6. High-level architecture

The system has two distinct lifetimes:

1. the **logical agent**, whose identity and durable state persist across restarts;
2. the **agent service**, which is one running process/session of that agent.

The human interface is a client, not the agent itself.

```text
                         +----------------------+
                         |       Clients        |
                         | CLI / TUI / web /    |
                         | future speech UI     |
                         +----------+-----------+
                                    |
                              local IPC/API
                                    |
                                    v
+----------------+       +----------------------+       +----------------+
| Environment /  | ----> |     Agent Service    | ----> | Delivery       |
| sensors/events |       |                      |       | adapters/outbox |
+----------------+       |  Event Queue         |       +----------------+
                         |       |              |
                         |       v              |
                         | Serialized Reducer   |
                         | only state writer    |
                         |    /        \        |
                         | commit    work item  |
                         +---|----------|-------+
                             |          |
                             v          v
                     +-----------+  +----------------+
                     |  SQLite   |  | Async workers  |
                     | durable   |  | LLM / embedder |
                     | agent     |  +-------+--------+
                     +-----------+          |
                                            | result event
                                            +----> Event Queue
```

The architecture is asynchronous end to end. The reducer never waits synchronously for an LLM call or local embedding computation.

The key concurrency invariant is:

> **Only the reducer commits agent state. Semantic workers consume immutable snapshots and return result events.**

The LLM is deliberately late in the pipeline. Cheap persistence, local indexing, activation, decay, candidate competition, and most gating happen without a generative LLM call.

The asynchronous architecture applies to cognition as well as conversation:

```text
event -> committed state evolves
           |
           +-> semantic work may be requested asynchronously
                        |
                        +-> result becomes a later event
```

Semantic worker results are never allowed to mutate state or contact the user directly. The reducer alone may create durable outbound intent. A delivery adapter may perform only that explicitly authorized transport side effect and must report the outcome back as a `DeliveryResult` event; it cannot mutate agent state.

### 6.1 Outbound delivery

A decision to speak and successful delivery are distinct states. The reducer persists the **decision/action** and an outbound item before any adapter contacts the user, but an undelivered item is not yet a user-visible conversational utterance.

```text
LLM/reducer decides SPEAK
        -> persist action
        -> persist outbound item: PENDING_DELIVERY
        -> delivery adapter attempts side effect
        -> DeliveryResult event
        -> DELIVERED / EXPIRED / FAILED / SUPERSEDED
```

Each outbound item has a stable `message_id` and unique `delivery_key`. Delivery adapters must carry the `delivery_key`; clients that can do so deduplicate by that key. Adapter execution may be at least once, while committed delivery state is effectively once.

Outbound items have two semantic classes:

- **MANDATORY** — generated to satisfy a response obligation; remains durable until delivered, superseded, or surfaced as an explicit failure.
- **PROACTIVE** — unsolicited conversational initiative; has a bounded delivery TTL and must be revalidated before delayed delivery. Stale proactive items expire rather than being dumped on the user later.

At most one undelivered proactive item should normally remain eligible per conversational channel. A newer proactive item may supersede or coalesce an older one. On client reconnect, mandatory items are considered first; after revalidation, at most one still-useful proactive item is delivered in a burst window.

Cooldown/refractory accounting for **human interruption** is anchored primarily to `delivered_at`, not merely to the earlier decision to speak. The self-model must not behave as if the human heard an item that expired or failed before delivery.

This keeps conversational intent independent of terminal lifetime while preserving timing semantics:

```text
close `aca chat`   != stop agent
stop agent service = suspend cognition
restart service    = resume same logical agent
```

Future embodiment replaces or augments delivery adapters (for example, TTS/speaker output) without changing cognition semantics.

### 6.2 Durable inbound delivery

Human input must survive connection uncertainty and process crashes. The client generates a stable `event_id` before sending a `HumanMessage`. The service acknowledges acceptance **only after the event is durably committed**.

```text
client creates event_id
      -> send HumanMessage
      -> service durably inserts event/inbox row
      -> ACK(event_id)
      -> reducer processes accepted event
```

If the connection fails before the ACK is observed, the client retries the **same** `event_id`. The durable event/inbox table enforces uniqueness, so ingress is at least once while logical event acceptance and reduction are effectively once.

Accepted-but-not-yet-reduced events survive restart and are replayed into the reducer before normal interactive processing. A daemon crash must not silently eat a human message that was already acknowledged.

## 7. State model

The logical agent maintains four broad categories of persistent state.

### 7.1 World state

What appears to have happened externally.

Examples:

- latest human messages;
- external events;
- timestamps;
- observable system state;
- current conversation activity.

### 7.2 User model

A lightweight evidence-based model of what appears relevant to the human.

Possible fields:

- recurring topics;
- explicit preferences;
- unresolved questions;
- engagement with previous agent initiatives;
- typical conversation times;
- recent interaction intensity.

The user model should avoid unnecessary psychological inference.

### 7.3 Self model

A model of the agent's own recent behavior and current tendencies.

Example:

```json
{
  "initiative": 0.43,
  "inhibition": 0.68,
  "recent_proactive_messages": 2,
  "last_outward_action_at": "2026-09-16T03:10:00Z",
  "dominant_topic": "asynchronous-conversational-agent-design",
  "proactive_cognitive_budget_remaining": 0.72
}
```

The self model supports metacognitive behavior such as:

- "I have spoken too much recently.";
- "I have already asked about this topic.";
- "This topic is interesting, but there is no reason to interrupt now.".

### 7.4 Conversation mode

Conversation mode is distinct from lifecycle state.

```text
ACTIVE
IDLE
DORMANT
```

| Mode | Meaning | Response expectation | Proactive behavior |
|---|---|---|---|
| `ACTIVE` | live back-and-forth conversation | short latency; silence only for naturally terminal/low-obligation turns | unrelated initiative strongly suppressed |
| `IDLE` | recent conversation, but no immediate turn expectation | moderate latency; delayed response may be natural | limited follow-up/reflection allowed |
| `DORMANT` | no active conversation | no reactive response expectation | stochastic autonomous initiative allowed |

Mode is inferred from recent cadence and turn structure, not only one timeout. A proactive message must not switch an unrelated `ACTIVE` conversation to another topic merely because an old memory became salient.

Mode is a coarse, cadence-based signal. It says *whether* a conversation is live, not *what it is about*. The v0.7 extension (§34) adds a **memory-anchored discourse focus** to conversation state — the id of the provisional memory of the current focus-setting turn — and, while mode is `IDLE`, enforces that a proactive message be compatible with that subject. So the finer promise above ("don't switch an unrelated live conversation to another topic") is enforced against a subject, not just a timeout. It is anchored to a *memory*, not a topic, because the memory exists synchronously while topic enrichment is asynchronous (see §34.3).

### 7.5 Identity and lifecycle state

Persist at least:

```text
agent_id
created_at
lifecycle_state
current_runtime_session_id
last_started_at
last_active_at
last_clean_suspend_at
last_resume_at
last_heartbeat_at
total_active_seconds
last_runtime_exit_kind
```

Logical lifecycle states for v0:

```text
RUNNING
SUSPENDED
RECOVERING
```

`INTERRUPTED` is recorded as an operational exit/recovery cause rather than as a durable identity break. If the process disappears without a clean suspension marker, the next runtime session records an unclean interruption and reconstructs the unavailable interval from the last durable heartbeat/commit.

### 7.6 Temporal state

The agent should have enough temporal context to reason about continuity without inventing experiences during downtime. Wall-clock chronology and **agent-observed availability** are distinct.

A semantic snapshot may include:

```text
current UTC time
current configured local time/timezone
wall_time_since_last_human_message
active_observed_silence_since_last_human_message
time since last DELIVERED agent message
time since last active cognition
current runtime-session age
most recent suspension/resume interval
wall time elapsed during that interval
```

`active_observed_silence_since_last_human_message` accumulates only while the lifecycle state is `RUNNING`. Time spent suspended, powered off, crashed, or recovering does not count as observed user absence.

Example:

```text
user speaks          Monday 10:00
agent runs until     Monday 12:00
host is off until    Friday 10:00

wall time since message      = 96h
agent-observed silence       = 2h
```

The semantic layer may know both facts: "we last spoke four days ago" and "I was unavailable for almost all of that interval." It must not infer prolonged user silence from the agent's own downtime.

Do not present downtime as remembered thought. The correct semantics are:

> **Real time passed; cognition was suspended.**

## 8. Event model and serialization

All inputs, lifecycle transitions, and asynchronous work completions enter through one event stream.

Core event types:

```text
HumanMessage
ExternalEvent
StochasticWake
ScheduledConstraint
AgentActionFeedback
AgentStarted
AgentSuspending
AgentResumed
RuntimeInterruptionDetected
EmbeddingResult
LLMResult
DeliveryResult
```

`MemoryReactivation` is normally represented as state selected during a cognition cycle rather than as a recursively emitted event.

Example human event:

```json
{
  "id": "evt_123",
  "type": "HumanMessage",
  "timestamp": "2026-09-16T07:20:00Z",
  "payload": {"text": "lol"},
  "source": "cli"
}
```

Events may update state even when no LLM call is made.

For client-originated `HumanMessage` events, `id` is an ingress idempotency key. A duplicate send with the same `event_id` is not a second conversational turn. The service ACKs the event only after durable acceptance; accepted events that were not yet reduced are recovered after restart.

### 8.1 Single logical writer

State transitions are serialized:

```text
human messages ------+
external events -----+--> event queue --> reducer --> commit --> next event
wake events ---------+
lifecycle events ----+
worker results ------+
```

The implementation may use concurrent async tasks, threads, or worker processes for network I/O, timers, local embedding inference, and LLM calls. Those workers never commit agent state.

The reducer owns a monotonically increasing `state_revision` and is the only component allowed to advance it.

The core transition is:

$$
S_{n+1} = R(S_n, E_n, A_n)
$$

where:

- $S_n$ is committed state before the event;
- $E_n$ is the current event;
- $A_n$ is an optional already-validated typed proposal carried by a result event;
- $R$ is the deterministic reducer.

### 8.2 Non-blocking semantic work

A cognition cycle may request semantic work without blocking the reducer.

The reducer commits the local transition first and persists a work item using an immutable snapshot:

```json
{
  "work_id": "work_81",
  "cycle_id": "cog_1001",
  "kind": "LLM_COGNITION",
  "basis_revision": 418,
  "source_event_id": "evt_123",
  "snapshot": "immutable-context-reference"
}
```

The worker eventually returns an `LLMResult` event. By then committed state may have advanced, so the reducer revalidates the result against current state before any semantic commit or creation of outbound intent.

This allows inbound human messages and lifecycle events to keep flowing while semantic work is in flight without abandoning serialized mutation semantics.

### 8.3 Supersession rule

If newer state invalidates a proactive result, the reducer may `DROP` or `DEFER` it. It must **not regenerate inline**.

Each logical cognition cycle may request at most one generative LLM call. A later independent event or stochastic wake may start a new cycle if the thought remains relevant.

### 8.4 Lifecycle events are state events

Startup, resume, suspension, and interruption detection use the same reducer/event semantics as conversation.

A lifecycle event may update temporal state and invalidate pending proactive work, but it must not directly synthesize missed cognition. In particular, `AgentResumed` is not permission to replay stochastic wakeups that would have happened while the service was unavailable.

## 9. Fast loop vs slow loop

The architecture uses a cheap structured loop and a rare semantic loop.

### 9.1 Fast loop

The fast loop is ordinary software plus optional **local embedding inference**. It is not a continuously running generative LLM.

A lightweight agent service remains resident while the agent is RUNNING so it can receive events and maintain timers.

Responsibilities:

- event intake and serialized reduction;
- eager raw-event persistence;
- cheap message classification;
- local embedding/index creation for eligible nontrivial human events;
- topic/provisional-memory activation and lazy decay;
- weighted candidate selection, including a null candidate;
- wall chronology, active-observed-silence, and conversation-mode tracking;
- hard budget/cooldown/capability gates;
- cheap staleness/TTL checks;
- stochastic wake scheduling;
- asynchronous work dispatch.

The fast loop should consume **zero generative-LLM tokens**. Local embedding inference has compute/storage cost, but does not consume the conversational LLM budget.

### 9.2 Slow loop

The slow loop invokes the generative LLM only when semantic interpretation is justified.

Responsibilities:

- semantic reflection;
- semantic enrichment of provisional memories;
- synthesis across retrieved memories;
- context-sensitive staleness/contradiction judgment;
- metacognitive interpretation;
- generation of an outward message when appropriate;
- typed proposals for persistent state changes.

Conceptually:

```text
FAST PATH
receive -> persist -> index -> reduce -> activate/decay -> sample -> gate
                                                        |
                                                        | rarely
                                                        v
SLOW PATH (async)
immutable snapshot -> LLM -> LLMResult event -> validate -> reducer -> optional output
```

Once a proactive candidate reaches the LLM, local gating should already imply a reasonably high probability of useful semantic work. LLM-selected silence remains possible, but a high LLM-called-but-silent rate indicates the local gate is too permissive.

---

## 10. Stochastic scheduler

A fixed schedule such as "think every 30 minutes" is intentionally avoided.

The first prototype uses a **piecewise-constant state-dependent internal activation hazard**. This controls opportunities for something to enter attention; it does **not** directly control whether the agent speaks.

At committed state $S_n$, compute:

$$
\lambda_{\text{think},n} = \lambda_0
f_{\text{observed-silence}}(S_n)
f_{\text{salience}}(S_n)
f_{\text{unfinished}}(S_n)
$$

Then sample:

$$
\Delta t \sim \mathrm{Exp}(\lambda_{\text{think},n})
$$

and schedule:

$$
t_{\text{wake}} = t_n + \Delta t
$$

`initiative`, `inhibition`, proactive cooldown, refractory recovery, and user quiet hours are deliberately absent from $\lambda_{\text{think}}$. They regulate **expression**, not whether an internal activation opportunity occurs.

Time-of-day may later be used as a resource-scheduling optimization for background compute, but that is an operational policy rather than simulated psychology. A configured quiet-hours promise must not reduce thought probability probabilistically; it must block unsolicited expression deterministically.

`f_{\text{observed-silence}}` must use agent-observed active time, not raw wall time since the last conversation. Downtime cannot masquerade as relational absence. The exact shape is an empirical v0 parameter: long observed silence may raise, lower, or saturate thought opportunity, but only time during which the agent was actually `RUNNING` may contribute to that signal.

Wall-clock time since the last conversation remains available as semantic context, but it is not itself evidence that the user ignored or drifted away from the agent.

This preserves the central distinction:

> **An inhibited agent may still have frequent internal activations while expressing very few of them.**

### 10.1 Resampling rule

Any committed event that materially changes wake-relevant state invalidates the previously sampled wake time.

```text
state commit
   -> cancel current wake timer
   -> evaluate lazy decay at current time
   -> recompute lambda_think
   -> sample next wake
```

Topic decay is evaluated lazily when an event is processed; the agent service does not need a periodic tick merely to decay values.

### 10.2 Why not exact non-homogeneous sampling in v0?

A mathematically exact time-varying hazard could use integrated-hazard inversion or thinning. That remains a valid later refinement.

For v0, piecewise-constant resampling provides:

- non-periodic timing;
- state dependence;
- cheap implementation;
- deterministic local replay when the RNG seed is captured;
- straightforward cancellation when new information arrives.

The RNG seed only reproduces the local stochastic core. Full replay additionally requires recorded worker results; live LLM providers are not assumed reproducible.

### 10.3 Time units

The canonical model time unit for rates and decay constants is **hours**.

Examples:

```text
spontaneous_activation_rate_per_hour
decay_rate_per_hour
refractory_tau_hours
```

Persist timestamps as RFC 3339 / UTC. Configuration may use human-readable durations such as `45m` or `2h`, but conversion to model units must be explicit.

### 10.4 Suspension and missed wakes

The stochastic scheduler exists only while lifecycle state is `RUNNING`.

If the service is suspended, powered off, or unavailable for an interval, stochastic wakeups that statistically might have occurred during that interval are **not replayed** later.

On resume:

```text
load committed state
  -> compute wall-clock elapsed time
  -> materialize wall-time decay / TTL / resolve current budget windows
  -> reconcile durable work
  -> record AgentResumed
  -> compute current lambda_think
  -> sample one fresh future wake
```

This is a hard semantic rule:

> **Time may pass without cognition.**

### 10.5 Clock sources

Use different clocks for different jobs:

```text
monotonic clock
  - in-process stochastic timers
  - request timeouts
  - worker timeout/renewal durations while process is alive

UTC wall clock
  - durable event timestamps
  - cross-restart elapsed time
  - deferred-intent TTL
  - last interaction chronology
  - fixed budget-window identity
  - persisted active-runtime/session accounting
  - persisted lease deadlines and cross-restart recovery
  - recovery and downtime calculation

configured local timezone
  - quiet hours
  - morning/evening interpretation
  - user-facing temporal language
```

Do not drive in-process timer durations directly from a mutable wall clock. Clock jumps, NTP corrections, DST, or timezone changes must not create phantom stochastic wakes.


### 10.6 Operational heartbeat and wake validation

Lazy decay requires no periodic cognition tick. The service nevertheless maintains one cheap **operational lifecycle heartbeat** so crash/interruption intervals can be reconstructed. The heartbeat is bookkeeping, not thought, and must not activate memories or invoke semantic cognition.

Each scheduled `StochasticWake` carries at least:

```text
runtime_session_id
scheduler_generation
scheduled_at_utc
```

Every reschedule/resume increments `scheduler_generation`. Before accepting a wake, the reducer verifies that:

- lifecycle state is still `RUNNING`;
- `runtime_session_id` is current;
- `scheduler_generation` is current;
- no host-suspension discrepancy occurred since it was scheduled.

This validation happens even if the timer callback was already queued by the runtime. An overdue callback that fires immediately after host resume therefore becomes a harmless stale event rather than a phantom cognition cycle. Timers submit events; they never have authority to cause cognition directly.

---

## 11. Outward-expression control

Internal activation and outward expression are separate stages.

Suppression mechanisms fall into two categories: **hard gates** and **soft expression factors**. They should not be described as one ordered ladder because the soft factors are combined rather than applied in precedence order.

### 11.1 Hard gates

Hard gates are boolean and short-circuit outward proactive speech.

Examples:

```text
capability allowed?
proactive budget available?
proactive cooldown clear?
conversation mode permits this initiative?
outside configured user quiet hours?
result not superseded by newer state?
compatible with the current discourse focus?  (v0.7, §34 — active only while mode is IDLE)
```

A hard proactive gate never suppresses an explicit user task merely because the autonomous/proactive budget is exhausted.

The discourse-focus gate (§34) is conditional: it applies only while conversation mode is `IDLE` and gates outward proactive **speech**, never candidacy — an unrelated thought remains a selectable candidate, it just isn't voiced into the wrong conversation. It is inactive during `DORMANT`, so autonomous resurfacing is preserved.

### 11.2 Proactive cooldown

A cooldown is a hard rule preventing repeated **unsolicited** messages inside a short window.

It does not prevent responding to an explicit user task.

### 11.3 Soft expression factors

Soft factors shape the probability that a valid thought becomes outward language. Their order is irrelevant.

Typical factors include:

- initiative temperament;
- inhibition temperament/adaptation;
- refractory recovery after recent unsolicited speech;
- salience;
- novelty;
- repetition pressure;
- conversational relevance.

A possible proactive expression model is:

$$
P(\text{speak}\mid\text{thought})=
\sigma(
  w_sS +
  w_nN +
  w_iI -
  w_hH -
  w_rR
)
$$

where:

- $S$ = salience;
- $N$ = novelty;
- $I$ = initiative;
- $H$ = inhibition;
- $R$ = refractory/repetition suppression.

### 11.4 Refractory recovery

After a **delivered** unsolicited message, proactive expression should temporarily become less likely and then recover smoothly.

One possible suppression term derives from:

$$
f_{\text{refractory}}(t)=1-e^{-t/\tau}
$$

where $t$ is hours since the last successfully **delivered** proactive message and $\tau$ is the configured recovery constant.

Refractory recovery affects **expression**, not the internal wake hazard.

### 11.5 Inhibition

`inhibition` is a slower-moving temperament/adaptation variable representing general conversational restraint.

For v0 it has one primary home: the optional/proactive **expression policy**. It is not also multiplied into $\lambda_{\text{think}}$.

This avoids double-counting the same temperament variable in both thought generation and speech suppression.

### 11.6 User quiet hours

Quiet hours are an optional **hard gate on unsolicited outward speech**. If the user configures a range such as `01:00-08:00`, proactive messages must not be emitted inside that interval in the configured local timezone.

Quiet hours do not suppress:

- internal stochastic activations;
- memory decay/reinforcement;
- responses to explicit user input;
- persisted background work that does not contact the user.

If the product later offers a separate overnight compute throttle, that belongs to runtime resource policy and must not be conflated with conversational quiet hours.

---

## 12. Memory, provisional memory, and topic activation

The memory system distinguishes **raw events**, **provisional memories**, and **semantically enriched topics**.

The governing rule is:

> **Persistence and cheap semantic addressability are eager; expensive semantic interpretation is lazy.**

### 12.1 Raw events

Every inbound human event is persisted before any semantic decision.

Raw persistence guarantees that silence never means data loss.

### 12.2 Provisional memories

Nearly all nontrivial human messages receive a cheap local provisional representation even when no generative LLM call occurs. Embedding eligibility must not depend on the `HIGH_INFORMATION` classifier label, because a classifier false negative would otherwise create a silent long-term recall gap.

V0 may skip embedding only obviously trivial content such as empty/whitespace input, a single reaction emoji, or a tiny allowlist of very short acknowledgements. The raw event is still persisted in every case.

Example:

```json
{
  "id": "pmem_42",
  "event_id": "evt_123",
  "text": "I wonder if machine agents can have a telos.",
  "keywords": ["machine", "agents", "telos"],
  "embedding_ref": "emb_42",
  "activation": 0.73,
  "salience": 0.81,
  "created_at": "...",
  "last_activated_at": "...",
  "decay_rate_per_hour": 0.08,
  "enrichment_status": "raw"
}
```

The local embedding exists to make the raw event semantically addressable later. It does not attempt to produce a rich natural-language interpretation.

This avoids a cold-retrieval failure in which a silent message can only be found again if a future wake happens to share its vocabulary.

### 12.3 Enriched topics

A semantically enriched topic may be represented as:

```json
{
  "id": "topic_42",
  "summary": "User is considering whether to return to fluid mechanics.",
  "tags": ["fluid-mechanics", "career", "research"],
  "activation": 0.71,
  "importance": 0.80,
  "unfinished": true,
  "created_at": "2026-09-15T22:12:00-03:00",
  "last_activated_at": "2026-09-16T01:42:00-03:00",
  "decay_rate_per_hour": 0.08,
  "source": "human_message"
}
```

Enrichment happens when:

1. a generative LLM call is already justified for the event, in which case topic extraction can be a byproduct; or
2. a provisional memory later wins attention strongly enough to justify a semantic enrichment call.

The system does **not** pay an LLM call merely because every substantive message arrived.

### 12.4 Lazy decay

Activation may decay approximately as:

$$
A(t)=A_0e^{-k\Delta t}
$$

where $k$ is measured per hour and $\Delta t$ is elapsed hours since the stored activation value was last materialized.

The database does not need periodic decay writes. Effective activation is computed when state is read or an event is reduced.

### 12.5 Persistence temperament

For unresolved topics and deferred intents, temperament `persistence` slows effective decay.

A simple v0 mapping is:

$$
k_{\text{effective}} = k_{\text{base}}(1-\alpha P)
$$

where $P\in[0,1]$ is persistence and $\alpha$ is a bounded global coefficient.

Persistence does not create new thoughts; it controls how long unresolved material remains competitive.

### 12.6 Reinforcement

Related events can raise activation.

Examples:

- the user mentions the same topic again;
- an external event relates to the topic;
- embedding retrieval finds a semantically similar provisional memory;
- a deferred response becomes newly relevant.

### 12.7 Stochastic selection with a null candidate

The candidate pool contains:

```text
enriched topics
+
deferred intents
+
provisional memories
+
NOTHING
```

The agent should not always pick the highest-activation item, and it should not be forced to pick anything.

A simple softmax-style distribution may use:

$$
P(T_i)=\frac{e^{z_i/T}}{e^{z_{\varnothing}/T}+\sum_j e^{z_j/T}}
$$

where:

- $z_i$ combines activation, importance/salience, novelty, unfinished status, and repetition penalties;
- $z_{\varnothing}$ is the score for `NOTHING`;
- $T$ is a selection temperature.

A provisional memory that wins selection may trigger semantic enrichment before any decision to speak. This is how silent-but-important messages can later become first-class topics without requiring eager LLM extraction.

---

### 12.8 Enrichment lifecycle, idempotency, and backoff

Semantic enrichment is bounded and idempotent. A provisional memory must not repeatedly consume generative budget merely because it remains salient.

Recommended lifecycle:

```text
RAW
  -> PENDING_ENRICHMENT
       -> ENRICHED
       -> FAILED -> backoff -> RAW
       -> DO_NOT_ENRICH   (after bounded failures or explicit semantic decline)
```

Persist at least:

```text
enrichment_status
enrichment_attempts
next_enrichment_after
last_enrichment_error
```

Rules:

- claiming enrichment must be idempotent by provisional-memory/work-item identity;
- a memory in `PENDING_ENRICHMENT` is not eligible for a second concurrent enrichment job;
- transient failure returns the memory to `RAW` only after bounded backoff;
- after a small configured attempt cap, transition to `DO_NOT_ENRICH`;
- `DO_NOT_ENRICH` does **not** delete or hide the memory: raw text, lexical index, and local embedding remain retrievable;
- a later explicit human event may create a new enriched topic without resetting the failed autonomous-enrichment loop.

This prevents a high-salience provisional memory from becoming a slow recurring token leak.

## 13. Human messages are stimuli, not RPC calls

Human messages always alter perceived conversational state, but not every message requires outward language.

The critical distinction is between **obligatory task-oriented input** and **optional conversational input**.

### 13.1 Direct tasks and task-intent questions

Explicit commands and genuine task requests bypass stochastic silence.

Examples:

```text
"Explain this stack trace."
"Rewrite this function."
"What does this error mean?"
"Find the bug in this code."
```

Policy:

```text
DIRECT_TASK / TASK_QUESTION
    -> deterministic response obligation
    -> stochastic silence disabled
```

The response may still fail because of a real runtime/provider error, but the agent must not intentionally ignore the request as personality behavior.

### 13.2 Classification asymmetry

Misclassification costs are asymmetric:

$$
C_{\text{miss task}} \gg C_{\text{answer unnecessarily}}
$$

Therefore ambiguous task/social messages are deliberately biased toward the response-required path.

```text
confident task          -> RESPONSE_REQUIRED
ambiguous task/social   -> RESPONSE_REQUIRED
confident social/ack    -> OPTIONAL_RESPONSE
```

The system should prefer an unnecessary answer over silently dropping a plausible task.

### 13.3 Optional social/informational turns

Low-obligation input may use stochastic response selection.

```text
Human: "lol"
Agent: <silence>
```

```text
Human: "I finally got the prototype running."
Agent: <may respond, may only update state>
```

Substantive optional turns may be persisted as provisional memories even when the agent stays silent.

### 13.4 Re-prompt recovery

A cheap in-band detector protects against task false negatives.

If the agent intentionally stayed silent and the user quickly sends a likely re-prompt, the new event is forced onto the response-required path.

Examples:

```text
"?"
"hello?"
"did you see that?"
near-duplicate repeat
rephrased version of the previous message
```

Useful signals include:

- recent unanswered human event;
- short re-prompt tokens such as `?`;
- high lexical similarity;
- high local-embedding similarity;
- repeated imperative/question structure.

A re-prompt recovery is also logged as evidence of a possible prior classification failure.

### 13.5 Delayed thought

A non-task statement or open-ended reflection may be remembered without an immediate response:

```text
Human: "I wonder whether machine agents can have a telos."
Agent: <silence>

later, after a valid stochastic wake and semantic relevance check:
Agent: "I've been thinking about what you said about telos..."
```

The later message is not scheduled at a fixed delay. A provisional memory, enriched topic, or deferred intent must win future stochastic selection and pass current-context validation.

### 13.6 Conversation-mode interaction

Response policy depends on liveness:

```text
ACTIVE:
  direct task/question -> respond
  substantive social turn -> usually respond
  acknowledgement/closer -> silence allowed
  unrelated proactive topic -> suppress

IDLE:
  direct task/question -> respond
  optional social turn -> stochastic
  delayed follow-up -> allowed when relevant

DORMANT:
  no reactive response expected
  autonomous initiative -> stochastic
```

Conversation mode is revalidated before proactive outbound intent is created, and delayed proactive items are revalidated again immediately before delivery.

---

## 14. Response obligation vs response desire

The system tracks two distinct ideas:

- **obligation**: whether conversational/task semantics require a response;
- **desire**: how strongly the current conversational state tends toward optional expression.

Obligation is not a coefficient in one global sigmoid.

### 14.1 Mandatory path

If classification or re-prompt recovery establishes a response obligation, stochastic silence is disabled.

```text
if response_required(event):
    create response obligation
    dispatch bounded response work
```

The obligation remains auditable until satisfied, explicitly superseded, or surfaced as a visible runtime failure.

### 14.2 Optional reactive path

Optional social turns may use a stochastic response model such as:

$$
P(\text{respond})=\sigma(
  w_dD +
  w_rR +
  w_iI -
  w_hH
)
$$

where:

- $D$ = response desire;
- $R$ = conversational relevance;
- $I$ = initiative/expressiveness tendency in the current context;
- $H$ = inhibition/repetition pressure.

Conversation mode changes the policy bounds. In `ACTIVE` mode, substantive turns should have much higher response probability than equivalent turns in `IDLE` mode.

### 14.3 Proactive path

For unsolicited speech, expression is evaluated only after a thought/candidate exists and semantic work is complete.

Internal activation does not imply outward initiative.

---

## 15. Silence as a first-class action

Silence is not an error state.

Important distinction:

```text
silence != discard input
```

For an optional human turn, silence may still produce:

- raw event persistence;
- cheap classification;
- provisional keyword/entity indexing;
- a local embedding;
- activation or salience changes;
- later lazy semantic enrichment.

A substantive silent message therefore remains retrievable even though no generative LLM call was paid at arrival time.

Intentional silence must be distinguishable internally from:

- provider failure;
- parser failure;
- timeout;
- budget rejection;
- application crash;
- stale/superseded model output.

For direct tasks, intentional silence is not a valid terminal action.

---

## 16. Deferred responses

A deferred response is a persisted latent intention, not necessarily a frozen draft.

Recommended representation:

```json
{
  "id": "intent_17",
  "topic_id": "topic_42",
  "intent": "Revisit the user's question about machine telos.",
  "activation": 0.77,
  "activation_boost": 0.15,
  "created_at": "...",
  "expires_at": "...",
  "status": "pending",
  "draft": null
}
```

A deferred intent participates in later candidate selection and decays according to the same persistence-aware activation rules as unresolved topics.

### 16.1 Cheap pre-gates

Before spending a generative LLM call, ordinary code checks:

- hard TTL not expired;
- effective activation above the candidate floor;
- current conversation mode permits resurfacing;
- intent not already expressed/resolved;
- hard proactive gates permit semantic work/output.

### 16.2 Semantic staleness check

Questions such as these are semantic and belong in the slow loop:

- Has newer context contradicted the deferred thought?
- Would resurfacing it now be coherent?
- Has the human implicitly answered or invalidated it?
- Should the thought be regenerated, transformed, or dropped?

The LLM receives the deferred intent plus current compact context and may propose `SPEAK`, `DEFER`, or `SILENCE`.

The default is to **regenerate language when the thought resurfaces** rather than store a frozen social draft.

### 16.3 Pre-outbox and delivery-time revalidation

Even after the LLM proposes `SPEAK`, the reducer re-checks current state before creating a proactive outbound item. If the conversation became `ACTIVE`, a cooldown appeared, a newer human event superseded the thought, or another hard gate now blocks output, the message is dropped or deferred.

If delivery is delayed because no client/channel was available, the proactive item is revalidated again immediately before transport. It may expire, be superseded, or be coalesced rather than reaching the user late.

A stale result is never regenerated inline.

This revalidation covers the *race* where the human changes the subject **after** a cognition cycle began (`last_human_message_at > cycle start` ⇒ superseded). It does **not** cover a cycle that begins *within* the current focus yet selects a candidate unrelated to it — a fresh wake, nothing superseded. That orphan case is the subject of §34.

---

## 17. Self-monitoring and adaptation

The agent may adapt from the relationship between its own actions and later user behavior, but evidence quality matters.

### 17.1 Evidence hierarchy

Strong evidence:

```text
"Stop asking me about this."
"That reminder was useful."
explicit thumbs-down / thumbs-up style feedback
```

Moderate evidence:

```text
user meaningfully engages with a proactive message
user explicitly changes a preference
```

Weak evidence:

```text
user changes subject
short response to proactive message
re-prompt after an intentionally silent turn (classification-quality evidence)
```

Zero adaptation weight by itself:

```text
no response within an arbitrary time window
```

Non-response is especially noisy in an asynchronous system. It must not accumulate negative adaptation simply because the user was absent.

### 17.2 Interpreting engagement

"Meaningful engagement" is itself a semantic interpretation. The LLM may propose a typed observation:

```json
{
  "type": "USER_ENGAGEMENT_OBSERVATION",
  "target_action_id": "act_123",
  "classification": "substantive"
}
```

Deterministic policy maps that classification to a bounded evidence weight. The LLM does not choose numeric parameter deltas directly.

Example policy mapping:

```text
explicit_positive       -> strong positive
substantive_engagement  -> moderate positive
short/polite response   -> near-zero
no_response             -> exactly zero
explicit_negative       -> strong negative
```

### 17.3 Bounded adaptation

Adaptive variables have:

- configured baselines;
- hard minimum/maximum bounds;
- maximum delta per evidence event;
- decay/regression toward baseline.

A simple model is:

$$
x(t+\Delta t)=x_0 + (x(t)-x_0)e^{-k\Delta t}+\Delta_{\text{evidence}}
$$

where $x_0$ is the configured temperament prior.

### 17.4 Cold start

Temperament values are priors. Adaptation should ramp in gradually as explicit evidence accumulates.

For v0, adaptation may be disabled for an initial evidence window and then enabled with conservative deltas.

### 17.5 LLM role

The LLM interprets interaction semantics; code decides what those interpretations are allowed to change.

> **The model interprets; deterministic policy assigns authority and magnitude.**

---

## 18. Temperament and cognition parameters

The v0 parameter space should remain deliberately small and each parameter should have one primary mechanism.

Recommended initial configuration:

```yaml
cognition:
  spontaneous_activation_rate_per_hour: 0.25

temperament:
  initiative: 0.50
  inhibition: 0.50
  persistence: 0.50

timing:
  refractory_tau_hours: 2.0
  proactive_cooldown: 45m
  quiet_hours:
    enabled: false
    start_local: "01:00"
    end_local: "08:00"

memory:
  default_decay_half_life_hours: 24

budgets:
  proactive_llm_calls_per_day: 12
  proactive_messages_per_hour: 2
```

Primary wiring:

| Parameter | Primary mechanism |
|---|---|
| `spontaneous_activation_rate_per_hour` | baseline $\lambda_0$ for internal wake opportunities |
| `initiative` | increases probability that an eligible thought becomes unsolicited speech |
| `inhibition` | decreases probability of optional/proactive outward expression |
| `persistence` | slows decay of unresolved topics and deferred intents |
| `refractory_tau_hours` | controls recovery of proactive expression after recent unsolicited speech |
| `quiet_hours` | hard-gates unsolicited outward speech during the configured local-time interval |

The separation is intentional:

```text
cognition parameters  -> what gets an opportunity to enter attention
temperament parameters -> what tends to be expressed
memory parameters      -> what remains cognitively available
```

New dimensions should be added only when observed behavior demonstrates a missing degree of freedom.

---

## 19. Cheap local gating

The generative LLM should not decide whether every event deserves a generative LLM call.

For a spontaneous/proactive opportunity:

```text
stochastic wake
      |
      v
sample topic / deferred intent / provisional memory / NOTHING
      |
      +---- NOTHING ------------------------------> STOP
      |
      v
cheap semantic-worthiness gate
      |
      +---- low ----------------------------------> STOP
      |
      v
hard budget/capability/output-eligibility checks
      |
      +---- blocked and no enrichment need ------> STOP
      |
      v
async LLM cognition/enrichment
```

The local semantic-worthiness gate may combine:

- effective activation;
- importance/salience;
- time since last mention;
- novelty;
- unfinished status;
- repetition penalty;
- whether the candidate is still only provisionally enriched.

Expression variables such as initiative and inhibition are not part of the internal wake hazard. They may still affect whether it is worth paying for an LLM call whose only plausible value would be proactive speech.

### 19.1 LLM-called-but-silent rate

A first-class efficiency metric is:

$$
R_{\text{silent}}=
\frac{\text{proactive LLM calls that end in no outward action and no useful enrichment}}
     {\text{all proactive LLM calls}}
$$

The denominator should not punish a call that intentionally stays silent but produces valuable semantic enrichment. Observability should therefore distinguish:

```text
silent + useful enrichment
silent + no useful state change
```

A high rate of the second category means tokens are being spent too early.

---

## 20. Cheap human-message classification

The first implementation uses rules and heuristics before introducing a local classifier model.

Useful classes:

```text
DIRECT_TASK
TASK_QUESTION
SOCIAL_QUESTION
ACKNOWLEDGEMENT
CONVERSATION_CLOSER
STATEMENT
HIGH_INFORMATION
LOW_INFORMATION
REPROMPT
```

`DIRECT_TASK`, `TASK_QUESTION`, ambiguous task-like messages, and `REPROMPT` enter the response-required path.

Examples likely to take a zero-token social path:

```text
"lol"
"yeah"
"ok"
"fair"
"nice"
"makes sense"
```

### 20.1 Conservative task boundary

The classifier is intentionally recall-biased for tasks. Near the decision boundary, route to response-required.

The dangerous error is not "answered when silence would have been fine." It is "silently ignored a real task."

### 20.2 Missed-task auditing

The classifier's own labels cannot measure its false negatives. Therefore the system must audit a random sample of optional-path silent turns using human relabeling or an offline stronger-model review.

Track:

$$
R_{\text{task-miss}}=
\frac{\text{actual tasks routed to optional path}}
     {\text{audited actual tasks}}
$$

Target: as close to zero as practical.

Re-prompt recoveries are also counted as online evidence of possible task misclassification.

A local classifier model should be added only after logs provide a labeled set of heuristic mistakes that can justify and evaluate it.

---

## 21. Generative LLM boundary

When a generative LLM call is justified, the worker receives an immutable, compact snapshot.

Suggested context bundle:

```text
cycle/work identifiers
+
basis state revision
+
agent temperament/state snapshot
+
current source event or candidate
+
recent conversation window
+
retrieved enriched topics
+
retrieved provisional memories
+
deferred intent, if any
```

Avoid sending complete conversation history.

A typical call should fit into hundreds to low thousands of tokens rather than tens of thousands.

Prompt construction used for tests/debug replay must be deterministic given committed state and the explicit clock:

- stable field ordering;
- deterministic retrieval ordering and tie-breaking;
- canonical serialization;
- no implicit wall-clock reads;
- explicit timestamps from event/state;
- stable prompt-template versioning.

A prompt hash may be recorded to detect accidental prompt drift.

---

## 22. LLM action contract

A generative worker returns **typed proposals** and never writes persistent state or emits directly to the user.

> **Language-model output is data, never authority.**

Example proactive result:

```json
{
  "action": "speak",
  "message": "I've been thinking about what you said earlier...",
  "proposals": [
    {
      "type": "ENRICH_PROVISIONAL_MEMORY",
      "provisional_memory_id": "pmem_42",
      "topic_summary": "Question about machine telos and purpose."
    },
    {
      "type": "ADJUST_TOPIC_ACTIVATION",
      "topic_id": "topic_42",
      "delta": -0.20
    }
  ]
}
```

Other valid actions include:

```text
SPEAK
DEFER
SILENCE
ACKNOWLEDGE
```

### 22.1 Validation pipeline

Every result event passes through:

```text
worker result
   -> schema parse
   -> work/cycle identity check
   -> basis-revision check
   -> action validation
   -> proposal whitelist
   -> identifier validation
   -> numeric/size clamping
   -> current-state revalidation
   -> deterministic reducer
   -> optional durable outbound intent
```

Only whitelisted proposal types are accepted. Numeric deltas have hard caps. No proposal may create an immediate recursive cognition event merely because state changed.

### 22.2 Pre-outbox and delivery-time revalidation

Every unsolicited `SPEAK` is revalidated against **current committed state** before the reducer creates a proactive outbound item, not only against the worker snapshot. A delayed proactive item is revalidated a second time before transport.

Checks include:

- conversation mode still permits initiative;
- proactive cooldown still clear;
- proactive budget still available;
- no newer human event superseded the candidate;
- the intent has not already been expressed/resolved;
- capability/policy still permits output;
- the candidate is still discourse-compatible with the *current* focus while mode is `IDLE` (v0.7, §34.6) — a candidate that was output-eligible at dispatch (or dispatched fail-open before the focus was embedded) but is now an `ORPHAN` is dropped here, exactly as a semantic near-repeat is; the delayed-delivery re-check adds a new discourse pass in the proactive delivery path (§34.6 checkpoint 3).

If any pre-outbox check fails, the reducer drops or defers the message. If a delivery-time check fails, the pending proactive item expires, is superseded, or is coalesced. Neither path regenerates inline.

### 22.3 Parse/validation failure

For proactive work:

```text
parse/validation failure -> log -> no outward output -> STOP
```

For a mandatory response obligation, a bounded repair or visible runtime failure is preferable to pretending intentional silence occurred.

### 22.4 Model interpretation vs policy effect

The model may propose semantic labels such as:

```text
SUBSTANTIVE_ENGAGEMENT
TOPIC_RESOLVED
CURRENT_CONTEXT_CONTRADICTS_INTENT
```

Deterministic policy decides what those labels are allowed to change and by how much.

---

## 23. Persistence and retrieval

SQLite is sufficient for the first prototype. Durable persistence is the continuity boundary of the logical agent.

Suggested tables:

```text
agent_identity
runtime_sessions
lifecycle_events
events                 # durable inbox / event log
provisional_memories
embeddings
topics
conversation_turns
agent_state
deferred_intents
response_obligations
work_items
actions
outbound_messages
delivery_attempts
feedback
budgets
cognition_traces
```

`agent_state` contains the monotonically increasing `state_revision`. `agent_identity` contains the stable `agent_id`; process restarts must never silently mint a replacement identity.

### 23.1 Example: provisional memories

```sql
CREATE TABLE provisional_memories (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    text TEXT NOT NULL,
    activation REAL NOT NULL,
    salience REAL NOT NULL,
    decay_rate_per_hour REAL NOT NULL,
    embedding_id TEXT,
    enrichment_status TEXT NOT NULL,
    enrichment_attempts INTEGER NOT NULL DEFAULT 0,
    next_enrichment_after TEXT,
    last_enrichment_error TEXT,
    created_at TEXT NOT NULL,
    last_activated_at TEXT NOT NULL
);
```

### 23.2 Example: topics

```sql
CREATE TABLE topics (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    activation REAL NOT NULL,
    importance REAL NOT NULL,
    unfinished INTEGER NOT NULL DEFAULT 0,
    decay_rate_per_hour REAL NOT NULL,
    source TEXT,
    created_at TEXT NOT NULL,
    last_activated_at TEXT NOT NULL
);
```

### 23.3 Example: deferred intents

```sql
CREATE TABLE deferred_intents (
    id TEXT PRIMARY KEY,
    topic_id TEXT,
    provisional_memory_id TEXT,
    intent TEXT NOT NULL,
    activation REAL NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL
);
```

### 23.4 Retrieval

V0 retrieval uses a hybrid cheap index:

- SQLite FTS/BM25;
- keywords/tags;
- local embeddings over nearly every nontrivial human message/provisional memory;
- explicit topic links.

A separate vector database is not required for v0. At prototype scale, cosine similarity can be computed over a bounded candidate set or through a lightweight SQLite vector extension if convenient.

Embedding inference is local and asynchronous. Failure to embed must not lose the raw event; lexical retrieval remains a fallback.

### 23.5 Durable work items and crash recovery

Semantic workers are ephemeral execution tasks; work ownership is durable. Persist every dispatched semantic job before a worker starts it.

Recommended work-item state machine:

```text
PENDING
  -> RUNNING (leased)
       -> COMPLETED
       -> FAILED_TERMINAL

RUNNING
  -> PENDING when lease expires
```

Persist at least:

```text
work_id
kind
cycle_id
basis_revision
source_event_id
status
attempt_count
lease_until
created_at
completed_at
result_event_id
last_error
```

Worker execution is **at least once**; reducer commits are **effectively once**. Result handling is idempotent by `work_id` / `result_event_id`.

On service startup, reconciliation runs before autonomous dispatch:

1. reclaim expired `RUNNING` items;
2. requeue eligible `PENDING` work;
3. reconcile pending mandatory response obligations;
4. redispatch recoverable mandatory work;
5. surface permanently failed mandatory obligations instead of leaving them silently pending;
6. enqueue recovered results through the ordinary event queue.

A process crash must never be observationally equivalent to intentional conversational silence.

### 23.6 Agent identity and runtime sessions

A minimal durable identity record:

```sql
CREATE TABLE agent_identity (
    agent_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    lifecycle_state TEXT NOT NULL,
    current_runtime_session_id TEXT,
    last_started_at TEXT,
    last_active_at TEXT,
    last_clean_suspend_at TEXT,
    last_resume_at TEXT,
    last_heartbeat_at TEXT,
    total_active_seconds INTEGER NOT NULL DEFAULT 0,
    last_runtime_exit_kind TEXT
);
```

Each process execution receives a distinct runtime-session row. The previous session is closed cleanly when possible. If no clean-close marker exists, recovery records an unclean interruption using the last heartbeat/commit as the lower bound for the unavailable interval.

### 23.7 Conversation history and outbound messages

Human messages remain durable conversation records even when no immediate response is produced. Intentional silence never deletes the human turn.

A reducer decision to speak is stored durably in `actions` and `outbound_messages`, but an agent utterance becomes part of the **delivered user-visible conversation** only after successful delivery. An expired or failed proactive item remains auditable as an attempted/decided action without pretending the human heard it.

Recommended `outbound_messages` fields:

```text
message_id
delivery_key UNIQUE
action_id
kind                 # MANDATORY | PROACTIVE
channel
payload
status               # PENDING_DELIVERY | DELIVERING | DELIVERED | EXPIRED | FAILED | SUPERSEDED
created_at
expires_at           # normally bounded for PROACTIVE; policy-specific for MANDATORY
delivered_at
superseded_by_id
last_delivery_error
```

Delivery attempts are idempotent in committed state by `message_id` / `delivery_key` / attempt identity. Where the client/adapter supports idempotency, it must also deduplicate display/side effects by `delivery_key`.

Proactive messages have delivery-time semantics:

- they expire after a configured TTL;
- they are revalidated against current mode, quiet hours, lifecycle state, cooldown/refractory state, relevance, and supersession before delayed dispatch;
- stale pending proactive items may be coalesced or superseded by newer ones;
- reconnect must not dump a backlog of unsolicited messages.

Mandatory items are not silently expired merely because the client was absent. They are delivered, explicitly superseded, or surfaced as failures.

### 23.8 Durable inbound events

Client-originated events use a durable inbox/event-log contract. At minimum, the event store tracks:

```text
event_id UNIQUE
source
type
payload
accepted_at
reduced_at
```

The service sends `ACK(event_id)` only after durable insertion. Re-sending an existing `event_id` returns the same acceptance result and does not create a second turn. Reducer completion marks `reduced_at` in the same logical transaction as the corresponding state transition.

On startup, accepted events with no completed reduction are recovered before ordinary new interactive events. This gives client ingress at-least-once transport semantics with effectively-once logical processing.

## 24. Bounded cognition and deterministic mutation

One of the strongest invariants is:

> **Reflection must not recursively trigger unbounded reflection.**

Bad:

```text
reflection
  -> memory update
  -> memory changed event
  -> reflection
  -> ...
```

Required logical behavior:

```text
source event / stochastic wake
  -> one bounded local cognition phase
  -> at most one generative LLM work request
  -> local commit
  -> STOP processing source event

later:
LLMResult event
  -> validate/revalidate
  -> one bounded result commit
  -> optional outward action
  -> STOP
```

The result event finalizes semantic work from the original `cycle_id`; it is not permission to recursively start another semantic call.

A stale result may be dropped or deferred, never regenerated inline.

All persistent mutation is performed by the reducer. The LLM and embedding workers cannot write database state, budgets, timers, temperament, capabilities, or outward messages directly.

---

## 25. Resource, cooldown, and token budgets

The agent should have explicit hard limits.

Example:

```yaml
budget:
  proactive_llm_calls:
    max_per_hour: 2
    max_per_day: 12

  proactive_tokens:
    max_input_per_day: 50000
    max_output_per_day: 10000

  expensive_model:
    max_proactive_calls_per_day: 5

cooldowns:
  after_proactive_message: 45m
```

A token bucket or equivalent limiter enforces budgets independently of the stochastic scheduler and independently of the LLM's requested action.

Budget windows are anchored to wall-clock window identities (for example UTC/local-calendar day according to configuration) and **never bank unused capacity across downtime**. Resume resolves the current window; it does not replay or accrue quotas from missed windows. A refillable bucket may fill only to its configured capacity, never beyond it because the service was offline.

These limits govern **autonomous/proactive cognition**. A direct user task follows the ordinary reactive-service budget and must not be silently dropped because the proactive token bucket is empty.

The system may expose a derived variable such as:

```text
proactive_cognitive_budget_remaining = 0.31
```

This is a resource abstraction, not a simulated emotion.


---

## 26. Observability and replay

Every cognition cycle and async work item should be inspectable.

Recommended trace fields:

```json
{
  "cycle_id": "cog_1001",
  "source_event_id": "evt_123",
  "basis_revision": 418,
  "commit_revision": 419,
  "trigger": "StochasticWake",
  "conversation_mode": "DORMANT",
  "candidate_kind": "provisional_memory",
  "candidate_id": "pmem_42",
  "candidate_was_null": false,
  "llm_called": true,
  "llm_result_revision_seen": 423,
  "action": "silence",
  "useful_enrichment": true,
  "pre_outbox_invalidated": false,
  "tokens_in": 612,
  "tokens_out": 18,
  "duration_ms": 840,
  "rng_seed_fragment": "...",
  "prompt_hash": "..."
}
```

The system should make it easy to answer:

- Why did the agent speak or remain silent?
- Was silence intentional, stale-result suppression, or runtime failure?
- Which topic/provisional memory/deferred intent was activated?
- Was a worker result based on stale state?
- How many proactive results were invalidated before outbox creation or before delayed delivery?
- How many LLM calls produced useful enrichment without speech?
- How many messages were embedded locally?
- How much autonomous cognition cost in tokens and local compute?
- Which typed proposals changed state?
- Was an outbound item merely decided, actually delivered, expired, coalesced, or failed?
- Did any adapter retry a `delivery_key`, and was duplicate user-visible delivery suppressed?
- Were any client events retried by `event_id`, and were they reduced exactly once?
- How much wall-clock silence versus agent-observed active silence preceded a proactive action?

### 26.1 Replay semantics

A fixed RNG seed reproduces only local stochastic decisions.

Full deterministic scenario replay requires:

```text
initial database snapshot
+ ordered external events
+ deterministic clock
+ RNG seed
+ recorded EmbeddingResult / LLMResult events
+ deterministic prompt/retrieval construction
```

Live provider calls are not assumed reproducible, even at temperature zero.

During tests, worker results should be mocked or replayed from fixtures. Prompt hashes can additionally detect unintended prompt drift.

---

## 27. Evaluation

The experiment should measure system efficiency, semantic-memory quality, classifier safety, and interaction quality.

### 27.1 Quantitative metrics

Track:

- proactive messages per day;
- proactive generative-LLM calls per day;
- generative input/output tokens per day;
- local embedding inferences per day and average latency;
- percentage of wake opportunities selecting `NOTHING`;
- percentage of candidates rejected before generative inference;
- proactive LLM-called-but-no-useful-output rate;
- silent-but-usefully-enriched LLM rate;
- direct-task intentional-silence rate (**target: 0%**);
- audited task false-negative rate for optional-path silent turns;
- re-prompt recovery count/rate;
- active-mode optional-turn silence rate;
- idle-mode optional-turn silence rate;
- provisional-memory resurfacing rate;
- lazy semantic-enrichment rate and token cost;
- pre-outbox and delivery-time invalidation counts;
- stale-worker-result count;
- percentage of delayed responses;
- mean and median interval between proactive messages;
- duplicate-topic/repeated-question rate;
- percentage of proactive messages that receive semantically classified engagement;
- explicit positive/negative feedback rate;
- duplicate outbound delivery rate (**target: 0 user-visible duplicates where the adapter supports idempotency**);
- proactive outbox expiry/coalescing count;
- pending proactive backlog depth (target normally 0-1 per channel);
- inbound event retry/dedup count;
- acknowledged-but-unreduced event recovery count;
- wall-time vs active-observed-silence at proactive initiation;
- stale-wake rejection count after suspend/resume.

The generic metric "% of human messages receiving no response" is not sufficient by itself. Silence must be segmented by message class and conversation mode.

### 27.2 Missed-task audit

Regularly sample optional-path silent messages and relabel them independently.

This audit is required because a classifier cannot detect its own task false negatives from its original labels.

Online re-prompt detection complements this offline audit but does not replace it.

### 27.3 Human evaluation

After selected proactive or delayed messages, collect lightweight feedback:

- Was this interruption useful?
- Did the timing feel natural?
- Was the silence before this response natural?
- Was this repetitive?
- Would an immediate response have been better?
- Did the message feel contextually motivated?
- Did a silence feel intentional or merely broken?

### 27.4 Primary success criteria

The main experiment is not whether the system can message the human first. That is trivial.

The useful criterion is:

> **The agent should sometimes surprise the user by speaking, while still making sense retrospectively why it chose that moment and topic.**

A second criterion is:

> **Optional silence should sometimes feel appropriate rather than broken.**

A third criterion is:

> **Direct tasks should remain reliably responsive despite stochastic social behavior.**

A fourth criterion is:

> **Substantive messages that received no immediate response must remain semantically retrievable enough to resurface later for reasons beyond lexical coincidence.**

---

## 28. Prototype implementation decisions

The cognitive architecture is language-independent. V0 should remain a single-host application with one durable logical agent and one agent service.

Recommended shape:

```text
aca-agent service
  + asyncio event queue / serialized reducer
  + SQLite
  + lifecycle/session manager
  + cancellable stochastic wake timer
  + deterministic state revision counter
  + conservative rule-based task classifier
  + cheap local embedding model
  + asynchronous generative-LLM worker
  + durable outbox / delivery adapters

aca client
  + CLI chat/status/logs initially
  + local IPC to aca-agent
```

### 28.1 Reference implementation: Python

The v0 reference implementation should use **Python 3.12+** with `asyncio`.

Python is preferred for the reference prototype because the likely evolution includes:

- local embeddings and model experimentation;
- Raspberry Pi deployment;
- microphone/speech recognition and TTS;
- OpenCV/camera perception;
- ONNX/PyTorch/TensorFlow Lite integration;
- GPIO, serial, I2C/SPI, and robotics tooling;
- rapid experimentation with stochastic/semantic policies.

The serialized-writer invariant maps cleanly to one reducer coroutine consuming an `asyncio.Queue`. Async workers may run concurrently but only return events.

The reducer/event loop must never run heavy CPU-bound inference inline. Network-bound LLM calls use ordinary async I/O. Local embeddings or other CPU-bound inference must run in a process pool or a native inference runtime explicitly known to release the GIL and avoid event-loop stalls. `run_in_executor` with threads is sufficient only when the underlying native library releases the GIL.

The architecture does not rely on Python-specific semantics; another implementation language may replace the service later without changing the state/event contracts.

### 28.2 Client/service boundary

The first user interface is a CLI client, but the CLI does not own the agent process.

Suggested commands:

```text
aca service start
aca service stop
aca status
aca chat
aca logs
aca memories
aca topics
```

Local IPC may use a Unix-domain socket where available or a loopback transport on platforms that need it. The transport is an adapter, not part of cognition semantics.

### 28.3 Classifier: rules first

Start with conservative heuristics. Add a learned local classifier only after real misclassification logs provide an evaluation set.

### 28.4 Embeddings: local from v0

Local embeddings are part of v0 because lazy semantic memory depends on non-lexical retrieval.

Embed nearly every nontrivial human message, independent of `HIGH_INFORMATION` / `LOW_INFORMATION` classification. Only obviously trivial inputs may skip embedding. Cheap mechanisms should bias toward recall; expensive semantic enrichment remains selective.

The embedding model should be small enough to run cheaply on the host and stable/versioned so indexes can be rebuilt reproducibly.

### 28.5 Generative model: one provider/model first

Use one generative model/provider initially. Introduce cheap/strong model tiers only when cost and quality metrics show a concrete need.

### 28.6 Embodiment boundary

If the architecture is later moved to a Raspberry Pi robot, ACA remains the high-level asynchronous cognition layer.

```text
Raspberry Pi / Python ACA
  perception / memory / dialogue / high-level intent
                    |
              serial / CAN / IPC
                    v
MCU (e.g. ESP32)
  IMU / encoders / balance / motor PWM / emergency stop
```

Hard real-time stabilization and safety control must remain outside the LLM/ACA process.

No agent framework, message broker, microservice split, spreading-activation graph, or multiple simultaneous latent-thought engine is required for v0.

## 29. Suggested v0.6 execution and lifecycle flow

The reducer never waits synchronously for semantic workers. Process lifecycle is part of the agent's durable event history.

### 29.1 Startup / resume

```text
1. Acquire an OS-owned single-instance advisory lock for this agent data directory / agent_id.
     - if already held: fail fast with a clear "agent already running" error;
     - process death must release the lock automatically; do not rely on a stale PID file.
2. Open the database.
3. Create a new runtime_session_id and scheduler_generation.
4. Inspect previous runtime-session close marker and last heartbeat.
5. Classify the previous unavailable interval:
     CLEAN_SUSPEND / HOST_RESTART / UNCLEAN_INTERRUPTION / UNKNOWN.
6. Enter RECOVERING.
7. Compute wall-clock elapsed time since last known active point.
8. Materialize wall-time-dependent decay, TTL expiry, cooldown expiry, and resolve current budget windows with no accrued credit.
9. Preserve agent-observed-silence semantics: downtime does not count as observed user absence.
10. Reclaim expired RUNNING work-item leases.
11. Requeue eligible PENDING semantic work.
12. Recover accepted-but-unreduced durable inbound events.
13. Reconcile every pending mandatory response obligation.
14. Redispatch recoverable mandatory work; surface terminal failures explicitly.
15. Expire/coalesce stale proactive outbox items; retain mandatory items according to obligation policy.
16. Record AgentStarted + AgentResumed/RuntimeInterruptionDetected as appropriate.
17. Set lifecycle state RUNNING.
18. Compute current lambda_think and sample exactly one fresh future stochastic wake under the new scheduler_generation.
19. Begin normal event processing.
```

**Do not replay stochastic wakeups from the unavailable interval.** The agent resumes with elapsed-time effects applied, not with fabricated missed cognition.

The single-instance lock is an OS/process-lifetime ownership mechanism (for example `flock` on Linux/Raspberry Pi or an equivalent named/file lock on Windows). SQLite transactions provide state durability; they are not held open for the entire process lifetime merely to act as a process mutex.

### 29.2 Inbound human/external event

For client-originated human input, ingress occurs before reducer processing:

```text
0. Client creates stable event_id and sends HumanMessage.
1. Service durably INSERTs the event/inbox row (or observes the same event_id already accepted).
2. Service ACKs event_id only after durable acceptance.
3. Event becomes eligible for reducer processing.
```

Reducer flow:

```text
4. Dequeue one accepted, not-yet-reduced event.
5. Materialize lazy decay at the event timestamp.
6. Infer/update conversation mode.
7. Classify the event conservatively.
8. Apply cheap local state changes; mark the event reduced in the same logical commit that advances state_revision.

9. If the human message is nontrivial under embedding eligibility:
     a. create/update provisional memory;
     b. persist embedding work item;
     c. dispatch local embedding work outside the reducer/event loop;
     d. do not wait.

10. If re-prompt detector fires:
     a. mark RESPONSE_REQUIRED;
     b. record possible prior classification miss.

11. If DIRECT_TASK / TASK_QUESTION / RESPONSE_REQUIRED:
     a. create auditable response obligation;
     b. build deterministic compact snapshot;
     c. persist at most one generative LLM work item for this cycle;
     d. dispatch asynchronously;
     e. persist trace;
     f. reschedule wake if wake-relevant state changed;
     g. STOP source-event processing.

12. Otherwise enter optional path:
     a. activate/reinforce topics, provisional memories, deferred intents;
     b. sample one candidate or NOTHING;
     c. if NOTHING -> trace/reschedule/STOP;
     d. apply cheap semantic-worthiness gate;
     e. apply hard proactive gates where relevant;
     f. if rejected and no enrichment reason -> trace/reschedule/STOP;
     g. build deterministic immutable snapshot;
     h. persist at most one generative LLM work item;
     i. dispatch asynchronously;
     j. trace/reschedule/STOP.
```

A duplicate client send with the same `event_id` never creates another human turn. If the service crashes after ACK but before reduction, startup recovery processes the accepted event later.

### 29.3 Embedding result

```text
1. Dequeue EmbeddingResult.
2. Validate work identity/model version and reject committed duplicates idempotently.
3. Attach embedding/index reference if the memory still exists.
4. Mark durable work completed and record result-event identity.
5. Advance revision and commit.
6. Do not recursively launch generative cognition merely because embedding completed.
7. Reschedule wake only if wake-relevant state changed.
8. STOP.
```

### 29.4 LLM result

```text
1. Dequeue LLMResult.
2. Validate work/cycle identity and schema; reject committed duplicates idempotently.
3. Compare basis_revision with current state_revision.
4. Validate and clamp typed proposals.
5. Re-evaluate current semantic/policy conditions.

6. For proactive/optional cognition:
     a. if superseded -> DROP or DEFER;
     b. never regenerate inline;
     c. if SPEAK -> run immediate pre-outbox revalidation;
     d. persist action + PROACTIVE outbound item with TTL/delivery_key only if current state permits;
     e. do not mark it as delivered or as a user-visible conversational turn yet.

7. For a mandatory response obligation:
     a. satisfy/generate the response item if result remains applicable;
     b. persist action + MANDATORY outbound item with stable delivery_key;
     c. otherwise mark obligation explicitly superseded/pending rather than intentional silence;
     d. never hide provider/parser failure as silence.

8. Apply accepted semantic/state proposals through reducer.
9. Mark durable work completed and bind result_event_id.
10. Advance revision and commit action/trace.
11. Cancel/resample wake if wake-relevant state changed.
12. STOP.
```

Delivery is a separately authorized side effect. Semantic workers never contact the user. The reducer creates durable outbound intent; delivery adapters perform only the authorized transport and return `DeliveryResult` events.

### 29.5 Graceful suspension / shutdown

On `aca service stop`, OS shutdown notification, or another graceful stop:

```text
1. Stop accepting new autonomous wake dispatch.
2. Cancel the in-process stochastic timer.
3. Record AgentSuspending.
4. Persist final heartbeat / active-time accounting.
5. Release or safely persist worker leases.
6. Mark runtime session cleanly closed with exit kind.
7. Set durable lifecycle state SUSPENDED.
8. Flush/commit SQLite transaction state.
9. Exit.
```

The logical agent remains the same agent while suspended.

### 29.6 Host sleep, crash, power loss, and kill -9

The process cannot always write a final suspension event. Persist a lightweight **operational heartbeat** and runtime-session metadata so the next startup can distinguish a clean close from an unclean interruption. This heartbeat is the one intentional periodic service tick; it is not cognition and cannot activate thoughts.

On recovery from an unclean stop:

- infer the unavailable interval from the last heartbeat/commit and current wall time;
- record `RuntimeInterruptionDetected`;
- treat the interval as cognition-free;
- do not add the unavailable interval to agent-observed user silence;
- recover durable mandatory work according to lease semantics;
- recover acknowledged-but-unreduced inbound events;
- never describe the interruption as intentional conversational silence;
- never mint a new `agent_id` solely because the process died.

Host suspend/resume where the same process survives is detected by a large discrepancy between wall-clock elapsed time and expected active/monotonic progress. The handler increments `scheduler_generation`, invalidates the pre-suspend timer, materializes elapsed wall-time effects, records `AgentResumed`, and samples a fresh wake.

A queued old timer callback is still harmless: every `StochasticWake` carries `runtime_session_id` and `scheduler_generation`, and the reducer performs suspension-gap validation before accepting it. If the wake predates a detected suspension/resume boundary, it is discarded.

### 29.7 Outbound delivery

Every outbound action is persisted before transport. Delivery then occurs through an authorized client/notification/TTS adapter.

```text
PENDING_DELIVERY
      -> delivery-time reducer revalidation
      -> DELIVERING / delivery attempt
      -> DeliveryResult
      -> DELIVERED | FAILED

PROACTIVE may also -> EXPIRED | SUPERSEDED
```

Delivery rules:

1. **Idempotency.** Every item has a stable `delivery_key`. Retried attempts reuse it. The adapter/client deduplicates by that key whenever supported. A duplicate `DeliveryResult` is idempotent.
2. **Mandatory priority.** On reconnect, valid mandatory responses are considered before proactive items. They do not silently expire merely because the UI was closed.
3. **Proactive TTL.** A proactive item has a bounded delivery TTL. If it is no longer timely, it expires without becoming a delivered conversational turn.
4. **Delivery-time revalidation.** Before dispatching a delayed proactive item, current mode, quiet hours, lifecycle state, relevance, cooldown/refractory state, and supersession are checked again.
5. **No dogpile.** Reconnect delivers at most one revalidated proactive item in the configured burst window. Other stale/similar proactive items are expired, coalesced, or superseded.
6. **Delivered-time semantics.** `delivered_at` is the primary timestamp for user-interruption cooldown/refractory and for claims such as "I already told the user." An item that never reached the user must not create false conversational memory.

Client absence or delivery failure is operational state, not conversational silence.

## 30. Safety and control invariants

These rules remain hard-coded outside the LLM:

1. **No unbounded recursive cognition.** A source cognition cycle requests at most one generative LLM call.
2. **Serialized mutation.** Only the reducer commits agent-state transitions and advances `state_revision`.
3. **Single service ownership.** Exactly one agent service may own a given local agent database/data directory at a time; an OS-owned advisory lock enforces this across process lifetime and is released automatically on process death.
4. **Workers are side-effect isolated.** LLM/embedding workers consume immutable snapshots and return result events only. Delivery adapters may perform only reducer-authorized transport side effects and cannot mutate agent state.
5. **LLM output is data, never authority.** Typed proposals are validated before they can affect state.
6. **All model-proposed deltas are whitelisted, bounded, and auditable.**
7. **No result may immediately re-trigger generative cognition merely because it changed state.**
8. **Stale/superseded proactive results are dropped or deferred, never regenerated inline.**
9. **Every proactive SPEAK decision is revalidated before outbox creation, and every delayed proactive item is revalidated again before delivery.**
10. **No bypass of proactive token, call, capability, cooldown, quiet-hour, lifecycle, or supersession gates.**
11. **Proactive gates never intentionally silence explicit user tasks.**
12. **Task-boundary classification is conservative; ambiguous task-like input routes to response-required.**
13. **Re-prompts after silence force response-required handling and are logged as possible missed-task evidence.**
14. **Durable ingress.** A client event is ACKed only after durable acceptance; retries reuse the same `event_id`; acknowledged-but-unreduced events survive restart and are reduced effectively once.
15. **No uncontrolled external side effects.** External actions require explicit capability/policy checks.
16. **No hidden permanent free-form chain-of-thought log.** Persistent introspective state is structured and inspectable.
17. **Intentional silence is distinguishable from provider failure, parse failure, timeout, stale-result suppression, delivery failure, budget rejection, suspension, crash, or expired proactive delivery.**
18. **Adaptive parameters have baselines, hard bounds, bounded deltas, and decay toward baseline.**
19. **User non-response has exactly zero negative adaptation weight by itself.**
20. **Deferred intentions require hard TTL plus cheap pre-gates and semantic current-context validation before resurfacing.**
21. **Raw events are persisted before optional semantic enrichment.**
22. **Nontrivial silent human events remain semantically addressable through local provisional indexing/embeddings independent of information-class labels.**
23. **Autonomous enrichment is bounded.** Concurrent duplicate enrichment is forbidden; failures use bounded backoff and a terminal no-auto-enrich state.
24. **Semantic work is durable.** Work is persisted before dispatch, leased, recoverable after crash, and result commits are idempotent.
25. **Mandatory response obligations survive process failure.** They are redispatched or surfaced as visible failures, never left silently pending forever.
26. **Quiet hours are a hard proactive-expression promise.** They do not probabilistically suppress internal cognition and do not block explicit reactive responses.
27. **Budgets do not bank across downtime.** Resume resolves the current fixed budget window; unused capacity from past windows never accrues.
28. **Replay claims are scoped correctly.** RNG seeds replay local stochastic decisions; full replay requires recorded worker results, a deterministic clock, and deterministic prompt/retrieval construction.
29. **Process uptime is not agent identity.** A restart creates a new runtime session, not a new logical agent.
30. **Suspension interrupts cognition, not identity.** Wall time may advance while active cognition remains zero.
31. **Agent downtime is not user absence.** Relationship/recency pressure may use agent-observed active silence, never downtime-dominated wall gaps as evidence of user disengagement.
32. **Missed stochastic cognition is never replayed after downtime.** Resume samples one fresh wake from current state.
33. **Clock semantics are explicit.** Monotonic time drives in-process durations; UTC wall time drives durable chronology/windows; active-runtime accounting drives observed availability; configured local time drives conversational time-of-day rules.
34. **Stale wake events are harmless.** A wake must match current lifecycle state, runtime session, and scheduler generation and survive suspension-gap validation before cognition starts.
35. **Heartbeat is operational, not cognitive.** The periodic lifecycle heartbeat cannot activate memories, alter conversational policy, or invoke semantic cognition.
36. **Clients do not own the agent lifecycle.** Closing a CLI/TUI/web client cannot implicitly stop or reset the agent service.
37. **Deciding to speak is not delivery.** An undelivered outbound item is auditable intent, not a user-visible utterance.
38. **Outbound delivery is idempotency-aware and time-sensitive.** Delivery keys suppress supported duplicates; proactive items have TTL/coalescing and cannot dogpile on reconnect.
39. **Heavy local inference is off the reducer/event loop.** CPU-bound embedding/perception work runs in a process pool or a native runtime known not to block the Python event loop.
40. **Hard real-time robot control is outside ACA.** Future embodiment delegates stabilization/safety loops to deterministic lower-level controllers.
41. **Discourse-focus gating is proactive-only, expression-only, and `IDLE`-scoped (v0.7).** It may suppress outward *proactive* speech unrelated to the current conversational focus **only while conversation mode is `IDLE`** (`ACTIVE` is already fully suppressed; `DORMANT` is left open so autonomous resurfacing is preserved). It runs only on proactive cycles and is revalidated before outbox and before delivery like any proactive SPEAK (invariant 9). It never removes a thought from candidacy, never blocks reactive or mandatory responses, and is inactive when mode is not `IDLE` or no focus vector exists. The gate **mechanism** (affinity, thresholds) is computed from stored embeddings by code with its own thresholds distinct from the observational cuts; the focus *subject* it scores against is set by the deterministic predicate (§34.4) or, on a call-making turn, by the v0.8 LLM focus transition (§35.3, invariant 42b) — the only LLM input to the focus.
42. **The LLM-proposed discourse relation is policy-gated, split by kind (v0.8).** (a) The **advancement relation** is *suppress-only*: it may narrow proactive speech (`ORPHAN`/`REPEAT` → not spoken), a missing/unknown label fails open to the §34 decision, and it can never force a speak the deterministic gates (§16.2, §34) would block nor be wired into the observational metrics. (b) The **focus transition** is a *validated state write* that moves the focus subject — thereby changing what the next gate suppresses (`CLEAR` fail-opens) — bounded to setting the focus only to a validated existing provisional-memory id (or `KEEP`/`CLEAR`) and writing no other state. Both ride a generative call that already happens (no new call) and are reproducible from recorded results on replay.

## 31. Open questions

The v0.6 core is frozen for implementation; §34 is the sanctioned v0.7 exception, admitted on prototype evidence per the rule below. Remaining questions are empirical tuning questions or accepted prototype limitations. Structural changes after this point require evidence from the running prototype.

### 31.1 Conversation-mode inference

What cadence/turn-structure heuristic best separates `ACTIVE`, `IDLE`, and `DORMANT` without making transitions feel mechanical?

### 31.2 Local gate calibration

What gate produces a low enough useless-LLM-call rate without filtering out surprising useful initiative or useful lazy enrichment?

### 31.3 Local embedding model

Which small embedding model gives adequate semantic recall at acceptable CPU/RAM cost on desktop and Raspberry Pi-class hardware, and how should embedding-version migrations rebuild the index?

### 31.4 Deferred-intent TTL

Should TTL depend on topic class, activation half-life, or one global conservative default in v0?

### 31.5 Memory consolidation

When should provisional memories and repeated observations merge into durable enriched topics?

### 31.6 Adaptation rate

How much explicit evidence is needed before temperament deviations from baseline become noticeable?

### 31.7 Circadian/topic shaping

Quiet hours are already a hard expression gate. Separately, should time-of-day influence topic selection or background-compute scheduling? It must not weaken the quiet-hours guarantee.

### 31.8 Exact scheduler refinement

Does the piecewise-constant hazard produce sufficiently natural timing, or does empirical behavior justify thinning / another non-homogeneous process?

### 31.9 Social silence policy

Within `ACTIVE` mode, which substantive non-task statements may naturally receive silence without making the system appear unresponsive?

### 31.10 Mandatory-response supersession and ordering

When multiple human messages arrive while a response worker is in flight, which newer turns supersede, merge with, or remain independent of an existing response obligation?

Accepted v0 limitation: independent mandatory workers may complete and their responses may be **delivered out of dispatch order**. V0 does not add head-of-line blocking solely to preserve ordering; traces must make ordering explicit.

### 31.11 Lifecycle heartbeat cadence

What heartbeat/session checkpoint interval best balances accurate interruption reconstruction with unnecessary writes? This is an operational tuning question; no heartbeat cadence may redefine logical identity.

### 31.12 Resume context horizon

For how long after resume should the latest suspension interval be included in semantic LLM context? The runtime always persists the lifecycle event; the open question is only how long it remains prompt-relevant.

### 31.13 Proactive delivery TTL and coalescing

What proactive outbox TTL and reconnect burst window feel natural in practice? These are delivery-tuning parameters; v0 must preserve the hard rule that stale proactive backlogs are not dumped on reconnect.

### 31.14 Observed-silence shaping

What shape should `f_observed-silence` use once there is real interaction data? Wall-clock downtime is excluded by definition; the open question is only how active observed silence affects thought opportunities.

### 31.15 Discourse-focus calibration (v0.7)

When the gate is active is **structural, not a tuning knob**: §34.7 ties it to `mode == IDLE`, so it always covers the `IDLE` orphan zone and never reaches `DORMANT`, for every accepted cadence config (the `IDLE` band is defined by the operator's existing `active_within`/`idle_within`). What remains genuinely empirical is the **affinity thresholds** separating `CONTINUE` / `BRIDGE` / `ORPHAN` (their own config keys, independent of the observational cuts — §34.5).

**Faithful focus transitions are an LLM-proposed relation — not another text heuristic — now specified in §35 (v0.8).** v0.7's focus predicate (§34.4) is a content-free structural approximation with known residuals (declarative subjects missed; confirmation questions mis-anchored) because "did this turn introduce/keep/drop the subject?" is not deterministically decidable. §35 realizes the **LLM-proposed `KEEP` / `REPLACE` / `CLEAR` relation** (deterministically gated per invariant 5) that subsumes the focus-setting predicate and, via the advancement relation (§35.4), the deferred `REOPEN` behavior. It was held until the deterministic gate's failure modes were measured against real traces (done 2026-09-22: the gate fires but advance-rate stays at switches), so the added non-determinism buys a measured improvement rather than a guess.

## 32. Future directions

Potential later extensions include:

- environmental event subscriptions;
- GitHub / file-system / calendar awareness;
- learned task/social classification;
- learned initiative policies;
- non-homogeneous hazard/thinning experiments;
- richer long-term memory consolidation;
- semantic association graphs / spreading activation;
- multiple simultaneous latent thoughts;
- multimodal perception;
- microphone/STT and TTS conversational adapters;
- Raspberry Pi deployment;
- physical embodiment with sensor events and high-level motion intents;
- multiple interacting agents;
- experimental comparison against ordinary request/response chat systems.

The v0 deliberately avoids association-graph activation spreading and multiple concurrent latent thought streams. One candidate competing against `NOTHING` is sufficient to test the core hypothesis.

For embodiment, ACA remains high-level and asynchronous. Physical safety, balancing, emergency-stop behavior, and other hard real-time loops belong to an MCU or equivalent deterministic controller rather than to the LLM agent service.

## 33. Core principles

The project should preserve these distinctions above all others:

> **A thought is not a message.**

> **Persistence and cheap semantic addressability are eager; expensive semantic interpretation is lazy.**

> **The model interprets; code decides what the interpretation is allowed to do.**

> **Process uptime is not agent identity; agent downtime is not user absence.**

> **Deciding to speak is not the same as being heard.**

> **Time may pass without cognition.**

The architecture should allow many internal activations to decay, compete, reinforce one another, or disappear without ever becoming language.

A substantive human message can become durable, semantically retrievable state without forcing either an immediate reply or an immediate generative LLM call.

The logical agent survives process restart through durable identity, memory, lifecycle history, and temporal state. Suspension does not fabricate experience: wall time advances, active cognition does not, and missed stochastic wakes are not replayed.

The generative LLM is used when semantic interpretation or linguistic expression is justified; it is not the clock that keeps the agent mentally alive and it is never the authority that mutates the agent directly.

In short:

```text
simulate the dynamics;
index experience cheaply;
interpret semantically only when justified;
let code retain authority;
persist identity across runtime sessions;
let time pass without inventing thoughts.
```

---

## 34. Discourse focus and topic continuity (v0.7)

### 34.1 The gap this closes

The v0.6 core gives the agent *semantic* continuity: memories and topics persist, decay, reinforce, and are retrieved by meaning (§12, §23.4). It does not give the agent *discourse* continuity — a model of **what conversation is happening right now** and whether a candidate thought belongs to it. The two are different, and the difference is visible in the running prototype: the agent voices thoughts that are individually worthwhile (activated, salient, novel, not recently expressed) but conversationally orphaned — dropped into a conversation about something else.

A representative live trace: the human moves the conversation to personal identity and continuity through sleep and coma; moments later the agent proactively emits an implementation observation about the embedding adapter. The thought is fine. Its *timing* is wrong. No v0.6 gate objects, because every v0.6 gate reasons about the thought in isolation, not about the conversation it lands in.

This section adds the missing abstraction: a lightweight **discourse focus** and one expression-stage gate that asks a question none of the existing gates ask —

> Is this thought appropriate to the conversation happening *now*?

### 34.2 Why the existing gates don't cover it

Three v0.6 mechanisms look adjacent but each solves a different problem:

- **Conversation mode (§7.4, §13.6)** is cadence-based. §13.6 already *states the intent* — in `ACTIVE`, "unrelated proactive topic → suppress" — but mode knows only *whether* a conversation is live, not *what it is about*, so v0.6 cannot enforce that intent against a subject. The `ACTIVE` window is also not brief: the prototype's `active_within` is minutes, and the mode gate suppresses *all* unsolicited speech in `ACTIVE` (`proactive_allowed_in_mode` requires `mode is not ACTIVE`). So the orphan cannot be an `ACTIVE` wake — it is an `IDLE` wake once cadence lapses, where mode permits initiative again but the human's subject is still the live one.
- **Pre-outbox / delivery-time revalidation (§16.3, §22.2)** catches the *race* where a human turn lands **after** a cycle began (`last_human_message_at > cycle start` ⇒ superseded). The orphan is not a race: it is a **fresh `IDLE` wake within the current focus** that selects an unrelated candidate. Nothing is superseded, so revalidation is silent.
- **The semantic-repetition mute (`continuity.py::is_semantic_repeat`, applied at the `wake.py`/`llm_result.py` expression checkpoints — §16.2 is a *different* thing, the LLM staleness check for deferred intents)** answers "have I already said this?" via deterministic cosine against recent/in-flight expressions. It says nothing about topical *fit*; a never-before-said thought about the wrong subject passes it cleanly.

The global candidate pool (`build_candidates`, §12.7) compounds this: every active topic, intent, and memory competes on intrinsic activation/salience with no notion of the current subject, and the only topicality signal in the reactive path is a coarse "is this the exact memory the human just created?" flag. Everything else is treated as equally on-topic.

### 34.3 Discourse focus state

Conversation state (§7.4 mode; §7.6 temporal state) gains one **discourse focus** field:

| Field | Meaning |
|---|---|
| discourse focus id | the id of the provisional **memory** of the current focus-setting human turn (the conversation's *subject*), or none |

The focus records only the *subject* ("what is this conversation about"). **Whether** the conversation is warm enough to enforce it is not a second timestamp — it is the existing conversation `mode` (§34.7). This is deliberate: keeping warmth on the one cadence signal `infer_mode` already computes avoids a second clock that can diverge from it.

The focus is anchored to the **provisional memory**, not a topic, deliberately: the memory of a human turn exists synchronously when the turn is reduced, whereas topic enrichment (§12.3) and its summary embedding are asynchronous and may not exist for seconds. The memory is the earliest durable, embeddable representation of "what the human just made this conversation about." (`ConversationState` is the implementing type; this schema is the DESIGN-level contract.)

An optional monotonic `discourse_epoch` may accompany this field for tracing, but it is not load-bearing in v0.7: the human-turn-after-cycle-start transition it would guard is already covered by supersession (§34.2).

### 34.4 Focus lifecycle

The focus is set by the reducer (single writer, invariant 2). In v0.7 that write is in the human-message handler at reduction time; in v0.8 (§35.3) a call-making turn writes a *provisional* focus here and its generative *result* may then override it (`KEEP`/`REPLACE`/`CLEAR`). The v0.7 deterministic transition turns on one question — **did this turn establish a new subject, i.e. produce a provisional memory from a focus-setting turn?** — with the pre-turn mode (evaluated before the handler updates `last_human_message_at`) deciding the fallback:

- **A focus-setting turn that produces a provisional memory** sets the focus to that memory id — a new subject, regardless of prior mode. In v0.7 a turn is *approximately* focus-setting via a **structural cue that introduces something to discuss** — a task/imperative or a question (including a short one, "what about a coma?"); shortness is not contentlessness. This is a separate axis from *response obligation*: a `REPROMPT` ("You there?") requires a reply but asserts no new subject, so it is **not** focus-setting; and a plain declarative statement is **not** focus-setting either (see the realization note below).
- **Any turn that does not establish a new subject** — a backchannel (`ok`/`noted`/…), *or* a focus-setting turn that created no memory (a terse-but-meaningful "no" that ingress drops as `is_trivial`) — does not revive or replace the subject on its own. It defers to the pre-turn mode:
  - **pre-turn `ACTIVE`/`IDLE`** (a live conversation): **keep** the existing focus. The conversation is still about the last substantive subject — the acknowledgement-train case — and holding it keeps warmth (refreshed cadence) and subject consistent.
  - **pre-turn `DORMANT`** (the conversation had lapsed): **clear** the focus (→ none). Nothing re-established a subject after the lapse, so yesterday's must not revive. With no focus the gate is inactive (§34.5 rule 1) until a substantive turn sets a new one.

The unifying invariant: **a subject survives a lapse into `DORMANT` only if a new substantive memory re-establishes one.** This closes both revival paths symmetrically — a bare `ok` *and* a no-memory "no" after dormancy each clear, not keep. It is also why warmth-follows-mode (§34.7) is safe: mode says *whether* we are in a conversation; the focus says *what it is about*, and the latter is revived only by substance.

**Realization: v0.7 focus-setting is a content-free *approximation* — a structural subject cue — not a faithful subject detector.** "Does this turn introduce a new conversational subject?" is a semantic question that no deterministic rule answers reliably; every cheap proxy has non-subject members:

- a *plain declarative* is undecidable vs an acknowledgement ("The robot needs better balance" vs "I hear you loud and clear") — the ingress class is only a length split (`HIGH_INFORMATION`/`STATEMENT`) and no word list closes the gap, so declaratives never set the focus;
- a *question or task* usually introduces a subject, but a **presence ping / confirmation check-in** ("You there?", "Is that clear?", "make sense?") carries only *response obligation*. The ingress class does not separate these — after a delivered agent turn "You there?" is a `TASK_QUESTION`, not a `REPROMPT` (which requires prior silence) — so both the `REPROMPT` class **and** recognized check-in *utterances* are excluded by a small, conservative phrase filter on the text. An *unlisted* check-in/confirmation phrasing remains a **residual** that can still be installed as the focus.

So v0.7 sets the focus on a task/question cue and never on a declarative, deliberately keeping focus-setting **separate from response obligation** (a mandatory reply is decided independently). Provisional-memory existence is not proof of substance — the predicate decides, and a memory only supplies the id once it has said focus-setting.

**Accepted v0.7 limitations (these are the contract, not hidden gaps). They fail in *both* directions — do not assume they only over-suppress:**

- **Declarative first subject → under-suppression (an orphan can be voiced).** A purely declarative opening subject is not caught, so there is no focus; the gate **fails open** (§34.5 rule 1) and an unrelated proactive candidate — an orphan relative to the human's actual (undetected) subject — **can be voiced**. This is the motivating failure the gate exists to prevent, and it is *not* caught when the subject was set declaratively.
- **Declarative mid-conversation shift → over-suppression.** The shift is missed, the prior subject is retained as stale focus, and an on-topic candidate for the new subject is wrongly `ORPHAN`ed (muted).
- **Unlisted check-in/confirmation phrasing → over-suppression.** Recognized presence/confirmation utterances ("You there?", "Is that clear?") are filtered out, but an *unlisted* variant ("was that understandable?") can still be installed as the focus, `ORPHAN`ing real candidates for one `IDLE` band.

All three reset on the next turn that carries a structural subject cue. In practice the focus anchors whenever the human asks a substantive question or gives a task (the motivating trace is question-driven), so the gate delivers value there while these residuals stand. A faithful predicate needs an LLM-*proposed* relation (deterministically gated, invariant 5); it is specified in **§35 (v0.8)** as a `KEEP`/`REPLACE`/`CLEAR` transition that supersedes this deterministic predicate on any human turn that makes a generative call, with this v0.7 predicate remaining the deterministic floor for turns that make none. Regression cases pin both the working paths and these documented residuals (§34.10).

**Agent speech does not move the focus** in v0.7. A voiced `BRIDGE` becomes, informally, the new thing the agent just made the conversation about, but the next wake is still scored against the last human focus memory. Letting agent output advance the focus is a deliberate later refinement, not part of the MVP.

### 34.5 Discourse relations and thresholds

Given a candidate and the current focus, the reducer computes an affinity from stored embeddings — the focus memory's vector vs the candidate's vector (a topic via its summary embedding, a memory via its own; §12.3, `continuity.py::candidate_vector`) — and classifies the relation by cosine band:

```text
CONTINUE : affinity >= discourse_continue_cosine                         direct continuation
BRIDGE   : discourse_bridge_cosine <= affinity < discourse_continue_cosine   a related move
ORPHAN   : affinity <  discourse_bridge_cosine                           no reason to say it now
REOPEN   : (deliberate return to a *different*, older thread — deferred, §34.8)
```

Two hard requirements on these thresholds:

1. **They are their own config keys** (`discourse_continue_cosine`, `discourse_bridge_cosine`) with their own v0 defaults (illustratively `0.75` and `0.55`, calibrated per §31.15). They **must not** default to, or be aliased onto, the observational cuts (`topic_dedup_cosine = 0.94`, `semantic_neighbor_threshold = 0.78`). If `discourse_bridge_cosine` were `0.78`, `ORPHAN` would become exactly the advance-rate "switch" class and shipping the gate would mechanically move the metric — the Goodhart coupling §17/§27 forbids. Sharing the *instrument* (cosine) is fine; sharing the *cut* is not. This is affinity to the *focus* (focus↔candidate), a different comparison than advance-rate's consecutive-speak cut anyway.
2. **In v0.7, `CONTINUE` and `BRIDGE` share the same expression fate** (both eligible). `discourse_continue_cosine` is therefore a **tracing label boundary only** in v0.7, not a gate boundary — the gate boundary is `discourse_bridge_cosine` (below it ⇒ `ORPHAN`). Giving `BRIDGE` distinct behavior (voiced-with-annotation, or deferred) is future work, not MVP.

**Vector-availability rules are three distinct cases** — fail-open is *not* the blanket rule (that is how the semantic layer went dark in prod, an empty `topic_embeddings`):

1. **Missing/cross-model *focus* vector ⇒ the gate is inactive** for that wake (candidate allowed, subject to the other gates). Blanket-silencing while embeddings lag is worse than a rare late orphan; the pre-outbox re-check (§34.6) catches the orphan once the focus vector lands.
2. **Missing/cross-model *candidate* vector ⇒ that candidate is not `ORPHAN`** (it is not suppressed), but other candidates are still scored normally. A candidate with no comparable vector is simply unjudgeable, not off-topic.
3. **A `DEFERRED_INTENT` candidate** has no vector of its own; it resolves to its `topic_id`'s summary embedding, or — simplest for v0.7 — deferred intents are out of scope of the discourse gate entirely. Either way it is never auto-`ORPHAN`ed for lack of a vector.

Consequence to state plainly: the motivating live orphan is a **topic** candidate, so it stays **uncaught until the topic-summary backfill (topic-embedding reconcile) has populated `topic_embeddings`** — the gate depends on that layer being real.

The relation is a per-decision structural signal computed from vectors — never the observational advance-rate or topic-dominance metric.

### 34.6 Where the gate sits: thought ≠ message, and its checkpoints

The gate changes expression, not cognition. "Candidacy untouched" means the off-topic thought **stays a candidate** — it is not removed from the pool and remains selectable on a later wake; it does **not** mean this cycle reinforces or generates it. It is **proactive-only**: it runs on `CYCLE_PROACTIVE` and is never consulted on a mandatory or optional-reactive cycle (invariants 11–13). It compares the **candidate vector** to the **focus vector** — an output-*eligibility* decision, not a judgment of a drafted sentence — so it can run before the generative call is spent.

A muted `ORPHAN` winner takes the **same path the continuity mute already takes** in `wake.py`, which is not uniform:

- if the candidate is **enrichment-eligible** (a RAW, salient provisional memory): it is dispatched **enrichment-only** (`output_eligible=false`, `LLM_ENRICHMENT`) — the memory is enriched but nothing is voiced;
- otherwise (a `TOPIC`, or a memory that fails the enrichment quality gate): `wake.py` returns **before** `_materialize_selected` and before creating work — the cycle ends at silence, **no activation is reinforced and no generative call is made**. The candidate simply remains stored for a future wake.

So the earlier phrasing "the thought is applied to state" holds only on the enrichment path; the topic-orphan path is plain silence. Both outcomes preserve the core distinction (§33): a thought is not a message.

Because a proactive `SPEAK` must be revalidated before outbox and again before delivery (invariant 9), the gate runs at more than selection. The semantic-repeat mute today implements **two** such checkpoints — wake pre-dispatch and `llm_result` pre-outbox; there is **no** existing delivery-time semantic check to mirror (`DeliveryPump` calls only the candidate-agnostic `evaluate_proactive`). The discourse gate matches those two and **adds new delivery-time plumbing**:

```text
1. wake.py                      set output_eligible BEFORE dispatch   (mirrors is_semantic_repeat)
2. llm_result.py pre-outbox     re-check vs CURRENT focus + vectors    (mirrors is_semantic_repeat)
3. DeliveryPump (NEW plumbing)  re-check the persisted outbound item's candidate_kind/candidate_id
                                against the current focus + vectors, before transport
```

Checkpoint 2 is not optional: the §34.5 fail-open *creates* the in-flight case — a wake with no focus vector yet dispatches output-eligible, the focus embedding lands, and by the time `LLMResult` arrives the candidate is now `ORPHAN`. Without the pre-outbox re-check the orphan is spoken after a slow LLM. Checkpoint 3 is genuinely new code (not a placement beside an existing check) and covers a proactive item that was on-topic at decision time but is delivered late, after the conversation moved on; it reads the outbound row's persisted `candidate_kind`/`candidate_id` (§23.7) rather than a live candidate object. Because all three are proactive-only expression checks, they belong beside `is_semantic_repeat` and in the proactive delivery path — **not** inside the candidate-agnostic `evaluate_proactive` and **not** in any shared helper the reactive path also calls. (§22.2's checklist is updated accordingly.)

```text
candidate selected              (unchanged; global pool)
    ↓
hard gates + semantic-repeat    (§11.1 — unchanged)
    ↓
discourse-focus gate            (§34 — new, IDLE-scoped, checkpoints 1–3)
    ↓
speak / enrichment-only / silence
```

**Accepted v0.7 silence consequence.** Because candidacy is untouched and parking is deferred (§34.8), when the global-pool winner is an `ORPHAN` in a warm conversation the cycle says **nothing** — it does *not* fall back to an on-topic runner-up that lost the sample. A hot off-topic topic that keeps winning softmax is muted on every warm wake while on-topic candidates go unspoken; the human can see silence in a warm conversation. This is a new, deliberate silence mode for v0.7. Removing it would require a bounded re-sample of the pool for an on-topic candidate — which *is* a candidacy change — and is therefore explicitly out of MVP scope.

### 34.7 When the gate is active (the load-bearing rule)

The discourse-focus gate is **active exactly while conversation mode is `IDLE`**. There is no separate `warm_window` parameter: warmth *is* `IDLE`, read from the mode `infer_mode` already computes. Outside `IDLE` the gate is inactive and expression falls back to the v0.6 gates unchanged.

Reusing mode is what makes the contract satisfiable and un-desynchronizable, which a separate window was not:

- **`ACTIVE`** (`gap ≤ active_within`): the mode gate already suppresses *all* unsolicited speech (`proactive_allowed_in_mode` requires `mode is not ACTIVE`), so the discourse gate is moot here.
- **`IDLE`** (`active_within < gap ≤ idle_within`): mode permits initiative; the discourse gate enforces affinity against the current focus subject — `CONTINUE`/`BRIDGE` may speak, `ORPHAN` is suppressed. **This is the orphan zone**, and it is exactly where the live-trace failure occurred (the prototype's `active_within` is short — 20s in the running config — so a proactive wake ~28s after a turn is already `IDLE`).
- **`DORMANT`** (`gap > idle_within`): gate inactive; autonomous resurfacing — "I've been thinking about the robot architecture from earlier…" — proceeds under the existing gates. This is the behavior the whole project exists to produce (§12.7, §16); a gate that reached into `DORMANT` would silence it, which is why the gate stops at the `IDLE`/`DORMANT` boundary.

`gap` is `now − last_human_message_at`, the single cadence signal mode already uses (UTC wall time, invariant 33). Because warmth and mode read the same signal, the two-clock divergence a separate window suffered from cannot occur:

- A **backchannel** refreshes `last_human_message_at` (so it keeps the conversation `ACTIVE`/`IDLE`, warm) but does **not** move the focus *subject* (§34.4). The gate therefore keeps enforcing affinity to the last *substantive* subject through an acknowledgement train — the case a `focus_set_at`-anchored window got wrong (there, focus went cold while the conversation was still live, re-admitting orphans).
- When the whole conversation lapses (no human turn for `idle_within`), mode itself becomes `DORMANT` and the gate turns off — the correct point to allow resurfacing again. Agent downtime that elapses `idle_within` likewise yields `DORMANT`, which is correct (invariant 31).

**Degenerate cadence configs are well-defined, not broken.** If an operator sets `idle_within == active_within` there is no `IDLE` band at all (mode steps straight from `ACTIVE` to `DORMANT`); the gate then simply has no zone in which it is active. That is a valid operator choice — the gate is off — not an unsatisfiable spec. (`config.py` already forbids `idle_within < active_within`.)

**Fail-open on a missing focus vector** (per §34.5 rule 1): while `IDLE`, when `focus_memory_id` has no vector yet, the gate does not fire for that wake; the orphan it might have let through is caught at checkpoint 2 (§34.6) once the focus vector lands.

### 34.8 Parked insights and REOPEN (deferred)

A thought that is cognitively strong but conversationally unfit now should not simply be discarded — it should become sayable *later*, when its subject is live again. The v0.6 `DeferredIntent` (§16) is the natural home: a high-salience `ORPHAN` may be parked as a deferred intent anchored to its own topic, and become expression-eligible when a future focus makes it `CONTINUE`/`BRIDGE`. When it does resurface it should be spoken as a *deliberate* return ("going back to something from earlier…"), which is the `REOPEN` relation.

Both parking and `REOPEN` are **deferred past the v0.7 MVP** on purpose: `REOPEN` is a new generative behavior with its own threshold and phrasing, and adding it before the basic on-topic gate is stable would compound two unproven mechanisms. v0.7 ships the `IDLE`-scoped `CONTINUE`/`BRIDGE`/`ORPHAN` gate; parking and `REOPEN` follow once its failure modes are measured.

### 34.9 Determinism, boundaries, and scope

- **Determinism / single writer.** The discourse focus (`focus_memory_id`) is written by the reducer: in v0.7, only in the human-message handler (at reduction); in v0.8, a call-making turn's generative *result* may also override it via the `llm_result` handler (§35.3, invariant 42b). Both are the same reducer (invariant 2 holds — a different handler writing durable state is fine); the "only the human-message handler" claim is v0.7-specific. Warmth is derived from mode (a function of `last_human_message_at`), and affinity is a pure function of stored vectors, so the decision replays under a recorded clock, stored embeddings, and recorded worker results (invariant 28). No new clock/RNG reads.
- **LLM is not authority (invariant 5).** v0.7's relation is computed from embeddings by code. Should an LLM later *propose* a relation (§31.15), it remains a proposal the deterministic gate may accept or reject — never the gate itself.
- **Cognitive vs conversational salience.** This section introduces a second, distinct notion of salience: a thought's intrinsic worth (its candidate score) is separate from its fitness for the current discourse (its affinity). "Remember it" and "say it now" are different decisions; §34 governs only the second.
- **Dependency.** Focus→memory affinity works today (provisional-memory embeddings are eager). Focus→**topic** affinity requires topic-summary vectors to be populated, so the gate is only fully effective once the summary-embedding backfill (topic-embedding reconcile) has run.

### 34.10 Regression cases

These pin the intended behavior and belong in the adversarial/counterfactual harness (§26, §27.2):

1. **Orphan suppressed while `IDLE`.** In an `IDLE` conversation whose focus is personal-identity/continuity, an embedding-adapter topic candidate (with its summary embedding present) is `ORPHAN` → stored, not spoken. (The live-trace failure.)
2. **Dormant resurfacing preserved.** In `DORMANT`, a low-affinity candidate still speaks — the `IDLE`-scoping must not silence autonomous resurfacing. *This is the guard on the extension itself: it fails if the gate is ever made unconditional.*
3. **Backchannel train keeps the gate on the substantive subject.** Substantive turn about X at t0; an `ok` backchannel at t1 (pre-turn `IDLE`) refreshes cadence but not the focus subject; a later `IDLE` wake with an unrelated candidate is still `ORPHAN` against X (not re-admitted). This is the sequence a `focus_set_at`-anchored window got wrong.
4. **Any no-new-subject turn after dormancy retires the subject.** Focus X set long ago; the conversation is `DORMANT`; a turn arrives that establishes no new subject-memory — whether a bare `ok`/empty turn (backchannel) **or** a terse-but-meaningful "no" classified focus-setting yet dropped as `is_trivial` (no memory). Either way, pre-turn mode was `DORMANT`, so the focus is **cleared** — a subsequent `IDLE` wake about an unrelated Y **speaks** (gate inactive, no focus), not suppressed against yesterday's X.
5. **A question sets the focus; acknowledgements, plain declaratives, and re-prompts do not.** A structural cue (`"what about the robots?"`) becomes the focus; `noted`/`alright`, an unlisted multiword ack (`"I hear you loud and clear"`), an ordinary declarative (`"The robot needs better balance."`), and a `REPROMPT` (`"You there?"` — obligation, not a subject) do **not**.
6. **Accepted v0.7 declarative limitation — both directions (documented contract).** A purely declarative *first* subject leaves the focus unset, so the gate **fails open and can voice an orphan** relative to the undetected subject (under-suppression). A declarative mid-conversation *shift* keeps the prior subject and may **over-suppress** an on-topic candidate for the new subject. Both asserted through `handle_human_message` so the contract — including the fail-open-voices-orphan direction — is pinned, not silently regressed.
7. **Focus-setting is separate from response obligation.** A substantive question both sets the focus and creates a mandatory obligation; a `REPROMPT` creates the obligation but not the focus. A `DIRECT_TASK` / re-prompt is answered while `IDLE` even if its content is `ORPHAN` to the focus (invariants 11–13).
8. **Optional-reactive untouched.** An optional social reply is unaffected by the discourse gate (proactive-only).
9. **Threshold independence.** A candidate whose affinity falls in the observational "switch" band is **not** automatically `ORPHAN` unless `discourse_bridge_cosine` independently places it there.
10. **Pre-outbox catch after fail-open.** Focus vector absent at wake ⇒ dispatched output-eligible; the focus embedding lands mid-flight; at `LLMResult` the candidate is now `ORPHAN` ⇒ dropped pre-outbox, not spoken (§34.6 checkpoint 2).
11. **Backfill dependency.** Before `topic_embeddings` is populated, a topic candidate has no vector ⇒ not `ORPHAN` (§34.5 rule 2); the gate only bites once the backfill has run.

---

## 35. LLM-proposed discourse relation (v0.8)

### 35.1 Why

Two measured limits of the v0.7 deterministic gate motivate this, and both are the *same* shape — a semantic judgment code cannot make:

- **Focus fidelity.** §34.4's focus predicate is a content-free structural approximation with documented residuals: a bare declarative subject is not anchored, a check-in question can be. "Did this turn introduce / keep / drop the subject?" is not deterministically decidable (§31.15).
- **Advancement.** Live measurement (2026-09-22): the discourse gate *fires* (`discourse_orphan` suppressions in the trace) and holds proactive speech on-focus, yet advance-rate stays at all-switches — the on-focus messages don't progressively *elaborate* a thread. "Is this message a real forward move on the thread?" is likewise semantic.

§34 deferred both here (§31.15) as an LLM-*proposed* relation. v0.8 realizes it.

### 35.2 The model

The LLM proposes a **discourse relation** — a typed label, the §22.4 pattern (model interprets; deterministic policy decides the effect) — at two points, each riding a generative call that already happens, so **no new call is added** (invariant 3). The two relations are **different kinds of thing**, and conflating them (as an earlier draft did) is a design error:

- the **advancement relation** on a proactive candidate (§35.4) is an **expression gate**. It can only *suppress* a speak (`ORPHAN`/`REPEAT` → not spoken), never force one — a claimed `ADVANCE` that is a cosine near-repeat is still muted by the continuity gate (§16.2), and a *missing* label falls back to the §34 decision, never silences. Suppress-only (invariant 42a).
- the **focus transition** on a human turn (§35.3) is a **state write**. It moves the *subject the gate scores against*, so by construction it changes *later* gating — `CLEAR` fail-opens the gate, `REPLACE` re-points it. It is therefore **not** suppress-only and must not be described as such; its safety is bounded differently (§35.6, invariant 42b): it may set the focus only to a *validated existing* subject and writes no other state, and it is the sole LLM input to the focus.

### 35.3 Focus transition (`KEEP` / `REPLACE` / `CLEAR`)

**Commit point — on the human-turn *result*, not at proactive pre-outbox.** When a human turn makes a generative call (all mandatory; an optional-reactive turn that actually drew a reply), the model returns the transition alongside its reply and the **`llm_result` handler for that human turn applies it**. This amends §34.4's "focus is written only in the human handler": the handler still writes a *provisional* focus via the deterministic predicate at reduction (so a wake before the result is scored against something), and the human-turn result then **overrides** it. Proactive pre-outbox (§22.2) would be the wrong place — a *silence* result never reaches it, and it is the speak revalidation, not a focus write.

- **`KEEP`** — focus unchanged (a check-in, an acknowledgement-with-a-question, an elaboration).
- **`REPLACE`** — the focus becomes a **validated provisional-memory id**: this turn's memory, or an earlier human turn's memory (a deliberate return). It is **not** a `topic_id` — §34.3 fixes the focus as a *memory* because that is what the gate resolves through `embeddings_by_memory`; a `topic_id` would validate as "existing", then miss the memory lookup and silently fail-open (§34.5 rule 1). An unknown/invalid id is treated as `KEEP`.
- **`CLEAR`** — focus becomes none; the gate then fails open until a new subject is set — a *deliberate* widening (§35.6), not a suppression.

**What it fixes, and what it does not.** It reliably fixes the *check-in* residual (`"You there?"` is `TASK_QUESTION` → makes the mandatory call → `KEEP` runs) and improves fidelity on every call-making turn. It does **not** fix the opening-*declarative* residual in the common case: a bare declarative (`"The robot needs better balance."`, `HIGH_INFORMATION`) usually takes `_optional_path` and **silences with no call** (`silence_nothing` / `silence_low_worth` / `silence_stochastic`), so no proposal exists and the v0.7 deterministic predicate (§34.4), which still refuses to anchor a declarative, governs. Closing that residual needs a **dedicated focus-classification call** on substantive no-reply turns — a *new* generative call — which is **deferred** (§35.7). v0.8 is a refinement layered on v0.7 for call-making turns; the deterministic path stays the floor everywhere else.

### 35.4 Advancement relation (the advance-rate lever)

A proactive generative result already returns a typed proposal (§22). v0.8 adds a required **relation** field classifying the drafted message against the current thread:

```text
ADVANCE   same thread, a new implication / consequence / decision   (the good one)
EVIDENCE  same thread, a new supporting fact or example
REVISE    same thread, a correction / qualification of an earlier claim
CLOSE     resolves or wraps the thread
REOPEN    a deliberate return to a *parked* older thread, framed as such
ORPHAN    unrelated to the current thread
REPEAT    restates something already said
```

Deterministic policy gates `SPEAK` **by suppression only**: `ORPHAN` and `REPEAT` degrade to enrichment-only or silence, at the **same pre-outbox checkpoint** and by the same mechanism §34.6 uses for a cosine orphan/near-repeat; every other value (the forward moves `ADVANCE` / `EVIDENCE` / `REVISE` / `CLOSE`, and `REOPEN` under a higher bar, §35.7) leaves the §34 decision untouched. This is the direct advance-rate intervention: a proactive utterance must *move* the thread, not merely sit near it.

**A missing or unknown relation fails open to the §34 decision — it is not a parse failure.** §22.3 (parse failure → no output) does **not** apply to the field's *absence*: a proactive result that omits `relation` — the current v0.7 worker, or any model that didn't emit it — is gated exactly as v0.7 (§34 cosine gate + continuity mute), never silenced for the omission. The advancement gate only *adds* suppression when the model *does* return `ORPHAN`/`REPEAT`; it is strictly suppress-only and backward-compatible.

### 35.5 Layering — the deterministic gates (§34) stay

The cheap deterministic gates remain the **pre-filter**, so a generative call is never spent on a clear orphan/repeat: the cosine discourse gate (§34) and the continuity mute (§16.2) still run at wake to set output-eligibility. The LLM relation is the **richer, authority-checked layer** at pre-outbox (§22.2), applied *after* the message is drafted, with the deterministic cross-checks still in force. Order: cheap deterministic pre-filter → generate → LLM-relation policy gate. Nothing in §34 is removed; v0.8 adds a semantic gate on top.

### 35.6 Invariants and boundaries

The two relations have **different safety envelopes** — invariant 42 is split accordingly:

- **Advancement relation — suppress-only (invariant 42a).** May only *narrow* proactive speech (`ORPHAN`/`REPEAT` → not spoken); a missing/unknown label fails open to the §34 decision (§35.4); it can never force a speak the deterministic gates (§16.2, §34) would block, and is never wired into the observational advance-rate/dominance metrics (Goodhart — else the agent moves its own metric by relabeling).
- **Focus transition — a validated state write (invariant 42b).** It moves the focus *subject*, which by design changes what the next gate suppresses — `CLEAR` fail-opens the gate (widening), `REPLACE` re-points it. It is bounded: it may set the focus **only** to a validated existing provisional-memory id (or `KEEP`/`CLEAR`), writes no other state, and is the **sole** LLM input to the focus. **This amends invariant 41:** §34's gate still computes affinity and thresholds *by code* (no LLM authority over the mechanism), but the focus subject it scores against may now be set by this transition.
- **One generative call (invariant 3):** both ride calls that already happen; **no call is added in v0.8**. (A future dedicated declarative focus call, §35.7, would be a new call and must be budgeted, invariant 10.)
- **Determinism / replay (invariant 28):** relations come from recorded worker results; replay reproduces both the gating and the focus write.

### 35.7 v0.8 MVP scope and deferred

- **MVP:** the focus transition applied on the **human-turn result** (§35.3, overriding the deterministic provisional focus), and the advancement gate at proactive **pre-outbox** (§35.4). A missing relation reproduces v0.7 behavior exactly.
- **Deferred:** the **dedicated focus-classification call** that would close the no-reply *declarative* residual (a new, budgeted generative call); `REOPEN`'s generative framing ("going back to something from earlier…") and parked-insight storage (§34.8); letting agent speech advance the focus; multi-thread state. In the MVP `REOPEN` is held to a high bar (or treated as `ORPHAN`) until parking exists.

### 35.8 Evaluation

Success = a measured rise in warm-zone advance-rate (fewer switches, more advances) against the v0.7 baseline (all-switches, 2026-09-22), with no regression in mandatory-answer latency or on-topic relevance. The counterfactual harness (§26, §27.2) and the advance-rate metric measure it; the LLM relation must never be wired into that metric.

### 35.9 Regression cases

These pin the subtle behaviors §35 introduces, at §34.10's level of specificity, so the follow-up implementation's tests are decided here rather than under implementation pressure:

**Advancement relation (suppress-only, invariant 42a):**

1. **`ORPHAN`/`REPEAT` suppress; forward moves pass.** A proactive result labeled `ORPHAN` or `REPEAT` is dropped at pre-outbox (enrichment-only or silence); one labeled `ADVANCE`/`EVIDENCE`/`REVISE`/`CLOSE` leaves the §34 decision untouched (it speaks iff §34 already allows).
2. **Missing/unknown `relation` ⇒ exact v0.7 behavior.** A proactive result that omits the field (the current worker) or emits an unrecognized value is gated by §34 alone — never silenced for the omission (it is *not* a §22.3 parse failure). This is the backward-compatibility guarantee.
3. **The label cannot force a speak.** A result labeled `ADVANCE` whose candidate is a cosine near-repeat is still suppressed by the continuity mute (§16.2) / an `ORPHAN` by §34 — a forward label never overrides a deterministic block.

**Focus transition (validated state write, invariant 42b):**

4. **`KEEP` / `REPLACE` / `CLEAR` effect.** On a call-making human-turn *result*: `KEEP` leaves the focus; `REPLACE(valid memory id)` sets the focus to that memory; `CLEAR` sets it to none — after which the next `IDLE` wake **fails the gate open** (a candidate the old focus would have `ORPHAN`ed can now speak), which is a *deliberate widening*, not a suppression (mirrors §34.10 case 4's deterministic `DORMANT` clear).
5. **`REPLACE` naming a non-memory id ⇒ treated as `KEEP`.** A proposed focus that is a `topic_id`, or any id that is not an existing provisional memory, is **not** adopted (it would validate as "existing" then miss `embeddings_by_memory` and fail-open, §34.5 rule 1) — the transition is treated as `KEEP`, not an error.
6. **Override commit point.** The transition is applied on the human-turn *result* (`llm_result`), overriding the provisional deterministic focus written at reduction; a wake between the human reduction and the result is scored against that provisional focus, not the (not-yet-arrived) proposal.
7. **No-call turn keeps the deterministic focus.** A human turn that makes no generative call — a backchannel, or a bare declarative that silences via `_optional_path` — carries no proposal, so the v0.7 deterministic predicate's focus stands (the documented declarative residual, §35.3).

**Boundaries:**

8. **No new generative call.** A human turn makes exactly the calls it made in v0.7, and a proactive cycle exactly one — the relations are fields on those results (invariant 3), and the LLM relation never appears in the advance-rate/dominance metric inputs (Goodhart).
