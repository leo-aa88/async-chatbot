# DESIGN.md — Asynchronous Conversational Agent

**Status:** Draft v0.4 — architecture frozen for v0 implementation  
**Scope:** Software-only prototype  
**Primary goal:** Build a persistent conversational agent that emulates human-like introspection through stochastic attention, memory activation, self-monitoring, inhibition, delayed response, silence, and autonomous initiative.

> **Architecture status:** Frozen for v0 implementation. New structural design changes should be driven by observed prototype behavior rather than speculative extensions.

---

## 1. Summary

Traditional chat systems are request/response machines:

```text
human input -> model -> response
```

This project deliberately breaks that contract.

The agent should behave more like an introspective human conversational partner:

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
5. **Maintain persistent conversational state** across long periods.
6. **Model its own recent behavior** well enough to regulate repetition, interruption, and conversational pressure.
7. **Use stochastic processes** so the same state does not always produce the same conversational behavior.
8. **Keep most autonomous cognition cheap** by running activation, decay, inhibition, and gating outside the LLM.
9. **Bound all cognition cycles** so internal reflection cannot recursively trigger uncontrolled reflection.
10. **Remain inspectable**: internal state should be structured and debuggable rather than hidden in a permanent free-form inner monologue.

### 2.2 Secondary goals

The architecture should make it possible later to add:

- external event sources such as GitHub, files, calendar, system state, or sensors;
- multimodal perception;
- local models for low-cost classification;
- multiple personalities or temperament profiles;
- physical embodiment in a robot;
- multiple cooperating agents.

---

## 3. Non-goals

The first version will **not** attempt to:

- claim or test machine consciousness;
- maintain a literal continuous linguistic stream of consciousness;
- continuously invoke an LLM while idle;
- simulate biological cognition faithfully;
- model emotions as if they were genuine subjective states;
- build a robot or embodied agent;
- provide a general autonomous-agent framework;
- use distributed infrastructure unless the prototype requires it;
- guarantee a response to every user message.

This project is a behavioral architecture experiment, not a theory of mind.

---

## 4. Terminology

### 4.1 Event

Anything that may alter the agent's state.

Examples:

- human message;
- stochastic wakeup;
- external system event;
- scheduled constraint;
- memory reactivation;
- explicit user command;
- feedback about a previous agent action.

### 4.2 Activation

A scalar representing how likely a topic, memory, or unresolved thread is to enter attention.

Activation is not language. It is structured state.

### 4.3 Reflection

Reasoning about an event, topic, memory, or unresolved issue.

Example:

> The user mentioned returning to fluid mechanics. This may be more than nostalgia.

### 4.4 Introspection / metacognition

Reasoning about the agent's own state or recent behavior.

Example:

> I have initiated three conversations today and the user ignored the last two; inhibition should increase.

The term **introspection** is used here operationally: the system emulates the behavioral pattern of introspective humans.

### 4.5 Initiative

The tendency to initiate outward communication without direct human prompting.

### 4.6 Inhibition

The tendency to suppress or delay an otherwise plausible outward response.

### 4.7 Cognition cycle

One bounded processing pass triggered by an event or stochastic activation.

A cognition cycle may end with:

- speech;
- deferred speech;
- state update only;
- silence.

---

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

The architecture is asynchronous end to end. The serialized reducer never waits for an LLM call or local embedding computation to complete.

```text
                       +----------------------+
                       |      Environment     |
                       +----------+-----------+
                                  |
                 +----------------+----------------+
                 |                                 |
          human messages                    external events
                 |                                 |
                 +----------------+----------------+
                                  |
                                  v
                       +----------------------+
                       |      Event Queue     |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | Serialized Reducer   |
                       | only state writer    |
                       +----+------------+----+
                            |            |
                    local commit         | async work request
                            |            v
                            |     +------------------+
                            |     | LLM / Embedding  |
                            |     |     Worker       |
                            |     +--------+---------+
                            |              |
                            |         result event
                            |              |
                            +--------------+
                                  |
                                  v
                       +----------------------+
                       | Persisted state +    |
                       | timers / actions     |
                       +----------------------+
```

The key concurrency invariant is:

> **Only the reducer commits agent state. Semantic workers consume immutable snapshots and return result events.**

The LLM is deliberately late in the pipeline. Cheap persistence, local indexing, activation, decay, candidate competition, and most gating happen without a generative LLM call.

The asynchronous architecture applies to cognition as well as conversation:

```text
event -> state evolves
           |
           +-> semantic work may be requested asynchronously
                        |
                        +-> result becomes a later event
```

A worker result is never allowed to mutate state directly or emit a message outside the reducer.

---

## 7. State model

The agent maintains three broad categories of persistent state.

### 7.1 World state

What appears to have happened externally.

Examples:

- latest human messages;
- external events;
- timestamps;
- observable system state;
- current conversation activity.

### 7.2 User model

A lightweight model of what appears relevant to the human.

Possible fields:

- recurring topics;
- explicit preferences;
- unresolved questions;
- engagement with previous agent initiatives;
- typical conversation times;
- recent interaction intensity.

The user model should remain evidence-based and should avoid unnecessary psychological inference.

### 7.3 Self model

A model of the agent's own recent behavior and current tendencies.

Example:

```json
{
  "conversation_state": "idle",
  "initiative": 0.43,
  "inhibition": 0.68,
  "recent_proactive_messages": 2,
  "recent_unanswered_proactive_messages": 1,
  "last_outward_action_at": "2026-09-16T03:10:00-03:00",
  "dominant_topic": "asynchronous-conversational-agent-design",
  "cognitive_budget_remaining": 0.72
}
```

The self model is the basis for metacognitive behavior such as:

- "I have spoken too much recently.";
- "I have already asked about this topic.";
- "The user tends not to engage with this kind of proactive question.";
- "This topic is interesting, but there is no reason to interrupt now.".


### 7.4 Conversation mode

The agent must distinguish whether it is currently in a live exchange or operating asynchronously.

```text
ACTIVE
IDLE
DORMANT
```

These modes change timing expectations and permissible silence:

| Mode | Meaning | Response expectation | Proactive behavior |
|---|---|---|---|
| `ACTIVE` | live back-and-forth conversation | short latency; silence only for naturally terminal/low-obligation turns | unrelated initiative strongly suppressed |
| `IDLE` | recent conversation, but no immediate turn expectation | moderate latency; delayed response may be natural | limited follow-up/reflection allowed |
| `DORMANT` | no active conversation | no reactive response expectation | stochastic autonomous initiative allowed |

Mode should be inferred from recent cadence and turn structure, not only one timeout. A simple v0 implementation may use time thresholds plus whether the latest human message creates an explicit response obligation.

A proactive message must not switch an unrelated `ACTIVE` conversation to another topic merely because an old memory became salient.

---

## 8. Event model and serialization

All inputs and asynchronous work completions enter through one event stream.

Core event types:

```text
HumanMessage
ExternalEvent
StochasticWake
ScheduledConstraint
AgentActionFeedback
EmbeddingResult
LLMResult
```

`MemoryReactivation` is normally represented as state selected during a cognition cycle rather than as a recursively emitted event.

Example human event:

```json
{
  "id": "evt_123",
  "type": "HumanMessage",
  "timestamp": "2026-09-16T04:20:00-03:00",
  "payload": {
    "text": "lol"
  },
  "source": "chat"
}
```

Events may update state even when no LLM call is made.

### 8.1 Single logical writer

State transitions are serialized:

```text
human messages ----+
external events ----+--> event queue --> reducer --> commit --> next event
wake events --------+
worker results -----+
```

The implementation may use concurrent goroutines for network I/O, timers, local embedding inference, and LLM calls. Those goroutines never commit agent state.

The reducer owns a monotonically increasing `state_revision` and is the only component allowed to advance it.

The core transition is:

$$
S_{n+1} = R(S_n, E_n, A_n)
$$

where:

- $S_n$ is the committed state before the event;
- $E_n$ is the current event;
- $A_n$ is an optional already-validated typed proposal carried by a result event;
- $R$ is the deterministic reducer.

### 8.2 Non-blocking semantic work

A cognition cycle may request semantic work without blocking the reducer.

The reducer commits the local transition first and creates a work item using an immutable snapshot:

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

The worker eventually returns a result event:

```json
{
  "type": "LLMResult",
  "work_id": "work_81",
  "cycle_id": "cog_1001",
  "basis_revision": 418,
  "proposals": []
}
```

By the time that result reaches the reducer, committed state may be at revision 423. The reducer therefore revalidates the result against current state before any commit or outward emission.

This allows inbound human messages to keep flowing while an LLM call is in flight without abandoning serialized mutation semantics.

### 8.3 Supersession rule

If newer state invalidates a proactive result, the reducer may:

```text
DROP
or
DEFER
```

It must **not regenerate inline**. Inline regeneration would turn staleness into an autonomous retry loop.

Each logical cognition cycle may request at most one generative LLM call. A later external event or stochastic wake may start a new cycle if the thought remains relevant.

---

## 9. Fast loop vs slow loop

The architecture uses a cheap structured loop and a rare semantic loop.

### 9.1 Fast loop

The fast loop is ordinary software plus optional **local embedding inference**. It is not a continuously running generative LLM.

A lightweight daemon remains resident so it can receive events and maintain timers.

Responsibilities:

- event intake and serialized reduction;
- eager raw-event persistence;
- cheap message classification;
- local embedding/index creation for selected substantive events;
- topic/provisional-memory activation and lazy decay;
- weighted candidate selection, including a null candidate;
- recency and conversation-mode tracking;
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
f_{\text{recency}}(S_n)
f_{\text{salience}}(S_n)
f_{\text{unfinished}}(S_n)
$$

Then sample:

$$
\Delta t \sim \operatorname{Exp}(\lambda_{\text{think},n})
$$

and schedule:

$$
t_{\text{wake}} = t_n + \Delta t
$$

`initiative`, `inhibition`, proactive cooldown, refractory recovery, and user quiet hours are deliberately absent from $\lambda_{\text{think}}$. They regulate **expression**, not whether an internal activation opportunity occurs.

Time-of-day may later be used as a resource-scheduling optimization for background compute, but that is an operational policy rather than simulated psychology. A configured quiet-hours promise must not reduce thought probability probabilistically; it must block unsolicited expression deterministically.

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

Topic decay is evaluated lazily when an event is processed; the daemon does not need a periodic tick merely to decay values.

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
```

A hard proactive gate never suppresses an explicit user task merely because the autonomous/proactive budget is exhausted.

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

After an unsolicited outward message, proactive expression should temporarily become less likely and then recover smoothly.

One possible suppression term derives from:

$$
f_{\text{refractory}}(t)=1-e^{-t/\tau}
$$

where $t$ is hours since the last proactive message and $\tau$ is the configured recovery constant.

Refractory recovery affects **expression**, not the internal wake hazard.

### 11.5 Inhibition

`inhibition` is a slower-moving temperament/adaptation variable representing general conversational restraint.

For v0 it has one primary home: the optional/proactive **expression policy**. It is not also multiplied into $\lambda_{\text{think}}$.

This avoids double-counting the same temperament variable in both thought generation and speech suppression.

### 11.6 User quiet hours

Quiet hours are an optional **hard gate on unsolicited outward speech**. If the user configures a range such as `01:00-08:00`, proactive messages must not be emitted inside that interval.

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

Conversation mode is revalidated again before any proactive message is actually emitted.

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

### 16.3 Pre-emit revalidation

Even after the LLM proposes `SPEAK`, the reducer re-checks current state immediately before emission. If the conversation became `ACTIVE`, a cooldown appeared, a newer human event superseded the thought, or another hard gate now blocks output, the message is dropped or deferred.

A stale result is never regenerated inline.

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
   -> optional outward emission
```

Only whitelisted proposal types are accepted. Numeric deltas have hard caps. No proposal may create an immediate recursive cognition event merely because state changed.

### 22.2 Pre-emit revalidation

Every unsolicited `SPEAK` is revalidated immediately before emission against **current committed state**, not only the worker snapshot.

Checks include:

- conversation mode still permits initiative;
- proactive cooldown still clear;
- proactive budget still available;
- no newer human event superseded the candidate;
- the intent has not already been expressed/resolved;
- capability/policy still permits output.

If any check fails, the reducer drops or defers the message. It does not regenerate inline.

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

SQLite is sufficient for the first prototype.

Suggested tables:

```text
events
provisional_memories
embeddings
topics
conversation_turns
agent_state
deferred_intents
response_obligations
work_items
actions
feedback
budgets
cognition_traces
```

`agent_state` contains the monotonically increasing `state_revision`.

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
- local embeddings over selected raw/provisional memories;
- explicit topic links.

A separate vector database is not required for v0. At prototype scale, cosine similarity can be computed over a bounded in-memory candidate set or through a lightweight SQLite vector extension if convenient.

Embedding inference is local and asynchronous. Failure to embed must not lose the raw event; lexical retrieval remains a fallback.

---

### 23.5 Durable work items and crash recovery

Semantic workers are ephemeral goroutines; work ownership is durable. Persist every dispatched semantic job before a worker starts it.

Recommended work-item state machine:

```text
PENDING
  -> RUNNING (leased)
       -> COMPLETED
       -> FAILED_TERMINAL

RUNNING
  -> PENDING when lease expires
```

Suggested fields:

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

Worker execution is **at least once**; reducer commits are **effectively once**. Result handling must therefore be idempotent by `work_id` / `result_event_id`. A duplicate result may be observed, but only the first accepted transition can satisfy an obligation or mutate semantic state.

On daemon startup, run reconciliation before normal autonomous dispatch:

1. reclaim `RUNNING` items whose lease expired;
2. requeue eligible `PENDING` work;
3. reconcile pending mandatory response obligations against surviving work/results;
4. redispatch recoverable mandatory work;
5. mark permanently failed mandatory obligations visibly failed rather than silently pending forever;
6. enqueue recovered results through the ordinary event queue.

A process crash must never be observationally equivalent to intentional conversational silence.

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
  "pre_emit_invalidated": false,
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
- How many proactive results were invalidated before emit?
- How many LLM calls produced useful enrichment without speech?
- How many messages were embedded locally?
- How much autonomous cognition cost in tokens and local compute?
- Which typed proposals changed state?

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
- pre-emit invalidation count;
- stale-worker-result count;
- percentage of delayed responses;
- mean and median interval between proactive messages;
- duplicate-topic/repeated-question rate;
- percentage of proactive messages that receive semantically classified engagement;
- explicit positive/negative feedback rate.

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

The first prototype should remain deliberately small.

Recommended shape:

```text
single Go daemon
  + serialized reducer/event queue
  + SQLite
  + cancellable stochastic wake timer
  + deterministic state revision counter
  + rule-based conservative task classifier
  + cheap local embedding model
  + asynchronous LLM worker
  + one generative LLM provider/model
  + CLI, TUI, or minimal web chat
```

### 28.1 Language: Go

Go is the default v0 choice because the prototype primarily needs:

- a long-lived single-binary daemon;
- clean timer cancellation/rescheduling;
- explicit concurrency boundaries;
- non-blocking worker dispatch;
- serialized state mutation;
- simple HTTP/API integration;
- low operational overhead.

### 28.2 Classifier: rules first

Start with conservative heuristics. Add a learned local classifier only after real misclassification logs provide an evaluation set.

### 28.3 Embeddings: local from v0

Unlike the generative model, local embeddings are part of v0 because lazy semantic memory depends on non-lexical retrieval.

Embed nearly every nontrivial human message, independent of `HIGH_INFORMATION`/`LOW_INFORMATION` classification. Only obviously trivial inputs may skip embedding. Cheap mechanisms should bias toward recall; expensive semantic enrichment remains selective.

The embedding model should be small enough to run cheaply on the host and stable/versioned so indexes can be rebuilt reproducibly.

### 28.4 Generative model: one provider/model first

Use one generative model/provider initially. Introduce cheap/strong model tiers only when cost and quality metrics show a concrete need.

No agent framework, message broker, microservice split, spreading-activation graph, or multiple simultaneous latent-thought engine is required.

---

## 29. Suggested v0.4 execution flow

The reducer never waits synchronously for semantic workers.

### 29.1 Inbound human/external event

```text
1. Dequeue one event.
2. Persist raw event.
3. Materialize lazy decay at the event timestamp.
4. Infer/update conversation mode.
5. Classify the event conservatively.
6. Advance state_revision and commit cheap local state.

7. If the message is nontrivial under the embedding-eligibility rule:
     a. create/update provisional memory;
     b. persist an embedding work item;
     c. dispatch local embedding work if needed;
     d. do not wait for embedding completion.

8. If re-prompt detector fires:
     a. mark RESPONSE_REQUIRED;
     b. record possible prior classification miss.

9. If DIRECT_TASK / TASK_QUESTION / RESPONSE_REQUIRED:
     a. create auditable response obligation;
     b. build deterministic compact snapshot;
     c. persist at most one generative LLM work item for this cycle;
     d. dispatch the persisted work item asynchronously;
     e. persist trace;
     f. reschedule wake if wake-relevant state changed;
     g. STOP processing this source event.

10. Otherwise enter optional path:
     a. activate/reinforce topics, provisional memories, deferred intents;
     b. sample one candidate or NOTHING;
     c. if NOTHING -> trace/reschedule/STOP;
     d. apply cheap semantic-worthiness gate;
     e. apply hard proactive gates as appropriate;
     f. if rejected and no semantic-enrichment reason -> trace/reschedule/STOP;
     g. build deterministic immutable snapshot;
     h. persist at most one generative LLM work item;
     i. dispatch the persisted work item asynchronously;
     j. trace/reschedule/STOP.
```

### 29.2 Embedding result

```text
1. Dequeue EmbeddingResult.
2. Validate work identity/model version and reject already-committed duplicate results idempotently.
3. Attach embedding/index reference to current raw/provisional memory if still present.
4. Mark durable work item completed and record result-event identity.
5. Advance revision and commit.
6. Do not recursively launch generative cognition merely because embedding completed.
7. Reschedule stochastic wake only if wake-relevant state changed.
8. STOP.
```

### 29.3 LLM result

```text
1. Dequeue LLMResult.
2. Validate work/cycle identity and schema; reject an already-committed duplicate result idempotently.
3. Compare basis_revision with current state_revision.
4. Validate and clamp typed proposals.
5. Re-evaluate current semantic/policy conditions as required.

6. If result belongs to proactive/optional cognition:
     a. if superseded -> DROP or DEFER;
     b. never regenerate inline;
     c. if SPEAK -> run immediate pre-emit revalidation;
     d. emit at most one message only if current state permits.

7. If result belongs to a mandatory response obligation:
     a. satisfy the obligation if result remains applicable;
     b. otherwise mark it explicitly superseded/pending rather than intentional silence;
     c. never hide provider/parser failure as silence.

8. Apply accepted semantic-enrichment/state proposals through reducer.
9. Mark durable work item completed and bind `result_event_id`.
10. Advance revision and commit action/trace.
11. Cancel/resample wake if wake-relevant state changed.
12. STOP.
```

A `StochasticWake` uses the same optional path, but begins with no human response obligation.

At no point does a stale worker result synchronously call the LLM again.

---

### 29.4 Startup reconciliation

Before accepting autonomous wake work after process start:

```text
1. Open database and acquire singleton daemon ownership.
2. Reclaim expired RUNNING work-item leases.
3. Requeue eligible PENDING semantic work.
4. Reconcile every pending mandatory response obligation.
5. Redispatch recoverable mandatory work.
6. Surface terminal mandatory failures explicitly.
7. Restore/cancel/resample the stochastic wake timer from current committed state.
8. Begin normal event processing.
```

Recovered worker completions enter through the same `EmbeddingResult` / `LLMResult` event path as live completions. Recovery never mutates agent state from a worker goroutine.

## 30. Safety and control invariants

These rules remain hard-coded outside the LLM:

1. **No unbounded recursive cognition.** A source cognition cycle requests at most one generative LLM call.
2. **Serialized mutation.** Only the reducer commits agent-state transitions and advances `state_revision`.
3. **Workers are side-effect isolated.** LLM/embedding workers consume immutable snapshots and return result events only.
4. **LLM output is data, never authority.** Typed proposals are validated before they can affect state.
5. **All model-proposed deltas are whitelisted, bounded, and auditable.**
6. **No result may immediately re-trigger generative cognition merely because it changed state.**
7. **Stale/superseded proactive results are dropped or deferred, never regenerated inline.**
8. **Every proactive `SPEAK` is revalidated against current mode, cooldown, budget, configured quiet hours, supersession, and capability immediately before emission.**
9. **No bypass of proactive token, call, capability, or cooldown gates.**
10. **Proactive gates never intentionally silence explicit user tasks.**
11. **Task-boundary classification is conservative; ambiguous task-like input routes to response-required.**
12. **Re-prompts after silence force response-required handling and are logged as possible missed-task evidence.**
13. **No uncontrolled external side effects.** External actions require explicit capability/policy checks.
14. **No hidden permanent free-form chain-of-thought log.** Persistent introspective state is structured and inspectable.
15. **Intentional silence is distinguishable from provider failure, parse failure, timeout, stale-result suppression, budget rejection, and crash.**
16. **Adaptive parameters have baselines, hard bounds, bounded deltas, and decay toward baseline.**
17. **User non-response has exactly zero negative adaptation weight by itself.**
18. **Deferred intentions require hard TTL plus cheap pre-gates and semantic current-context validation before resurfacing.**
19. **Raw events are persisted before optional semantic enrichment.**
20. **Nontrivial silent human events remain semantically addressable through local provisional indexing/embeddings independent of information-class labels.**
21. **Autonomous enrichment is bounded.** Concurrent duplicate enrichment is forbidden; failures use bounded backoff and a terminal no-auto-enrich state.
22. **Semantic work is durable.** Work is persisted before dispatch, leased, recoverable after crash, and result commits are idempotent.
23. **Mandatory response obligations survive process failure.** They are redispatched or surfaced as visible failures, never left silently pending forever.
24. **Quiet hours are a hard proactive-expression promise.** They do not probabilistically suppress internal cognition and do not block explicit reactive responses.
25. **Replay claims are scoped correctly.** RNG seeds replay local stochastic decisions; full replay requires recorded worker results and deterministic prompt construction.

---

## 31. Open questions

The architecture is frozen for the v0.4 implementation. Remaining questions are empirical tuning questions or explicitly accepted v0 limitations, not unresolved structural semantics.

### 31.1 Conversation-mode inference

What cadence/turn-structure heuristic best separates `ACTIVE`, `IDLE`, and `DORMANT` without making transitions feel mechanical?

### 31.2 Local gate calibration

What gate produces a low enough useless-LLM-call rate without filtering out surprising useful initiative or useful lazy enrichment?

### 31.3 Local embedding model

Which small embedding model gives adequate semantic recall at acceptable CPU/RAM cost, and how should embedding-version migrations rebuild the provisional index?

### 31.4 Deferred-intent TTL

Should TTL depend on topic class, activation half-life, or one global conservative default in v0?

### 31.5 Memory consolidation

When should provisional memories and repeated observations merge into durable enriched topics?

### 31.6 Adaptation rate

How much explicit evidence is needed before temperament deviations from baseline become noticeable?

### 31.7 Circadian/topic shaping

Quiet hours are already a hard expression gate. Separately, should time-of-day eventually influence topic selection or background-compute scheduling? This is not required for v0 and must not weaken the quiet-hours guarantee.

### 31.8 Exact scheduler refinement

Does the piecewise-constant hazard produce sufficiently natural timing, or does empirical behavior justify thinning / another non-homogeneous process?

### 31.9 Social silence policy

Within `ACTIVE` mode, which substantive non-task statements may naturally receive silence without making the system appear unresponsive?

### 31.10 Mandatory-response supersession and ordering

When multiple human messages arrive while a response worker is in flight, which newer turns supersede, merge with, or remain independent of an existing response obligation? V0 should keep this policy conservative and observable rather than silently dropping obligations.

Accepted v0 limitation: independent mandatory workers may complete and emit **out of dispatch order**. V0 does not introduce response sequencing/head-of-line blocking solely to preserve ordering; traces must make the ordering explicit.

---

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
- embodiment in a physical robot;
- multiple interacting agents;
- experimental comparison against ordinary request/response chat systems.

The v0 deliberately avoids association-graph activation spreading and multiple concurrent latent thought streams. One candidate competing against `NOTHING` is sufficient to test the core hypothesis.

---

## 33. Core principles

The project should preserve three distinctions above all others:

> **A thought is not a message.**

> **Persistence and cheap semantic addressability are eager; expensive semantic interpretation is lazy.**

> **The model interprets; code decides what the interpretation is allowed to do.**

The architecture should allow many internal activations to decay, compete, reinforce one another, or disappear without ever becoming language.

A substantive human message can become durable, semantically retrievable state without forcing either an immediate reply or an immediate generative LLM call.

The generative LLM is used when semantic interpretation or linguistic expression is justified; it is not the clock that keeps the agent mentally alive and it is never the authority that mutates the agent directly.

In short:

```text
simulate the dynamics;
index experience cheaply;
interpret semantically only when justified;
let code retain authority.
```
