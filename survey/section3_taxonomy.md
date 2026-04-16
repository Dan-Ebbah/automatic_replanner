# 3. A Taxonomy of Failure Modes in LLM Agent Systems

The reliability of LLM-based agents remains one of the most significant barriers to their deployment in high-stakes, production environments. While individual failure types — hallucination, tool misuse, infinite loops — have been studied in isolation, no unified taxonomy exists that (i) categorizes failures by their *origin*, *manifestation*, and *impact*, (ii) maps each category to appropriate detection and recovery strategies, and (iii) accounts for the multi-agent, workflow-level failures that emerge when agents are composed into pipelines.

This section proposes such a taxonomy. We organize failures along three orthogonal dimensions — **origin** (where the failure arises), **manifestation** (how it presents), and **severity** (how much of the system it affects) — and then enumerate seven concrete failure types that span this space. For each type, we characterize its detectability, typical root causes, and the class of recovery strategies it admits.

## 3.1 Dimensions of Failure

We identify three orthogonal dimensions along which agent failures can be characterized:

### 3.1.1 Origin: Where Does the Failure Arise?

| Origin Layer | Description | Examples |
|---|---|---|
| **Model-internal** | The LLM itself produces incorrect, incoherent, or unsafe output | Hallucination, refusal, mode collapse |
| **Interface** | The boundary between the LLM and external systems fails | Malformed tool calls, schema violations, parsing errors |
| **Environment** | External dependencies — APIs, databases, services — fail | Timeouts, rate limits, stale data, network errors |
| **Compositional** | Failures that emerge from agent *interaction*, not from any single agent | Cascading errors, deadlocks, goal divergence across agents |

This layered view is critical because **the appropriate recovery strategy depends on the origin layer.** A hallucination (model-internal) can be addressed by prompt repair or grounding injection, while a cascading failure (compositional) requires workflow-level recomposition — prompt-level fixes are insufficient.

### 3.1.2 Manifestation: How Does the Failure Present?

Failures can manifest in ways that range from immediately observable to deeply latent:

- **Explicit**: The system raises an exception, returns an error code, or violates a type contract. These are trivially detectable via conventional error handling.
- **Structural**: The output exists but violates expected formats — missing JSON fields, broken XML, truncated responses. Detectable via schema validation.
- **Semantic**: The output is well-formed and plausible but *wrong* — factually incorrect, off-topic, or misaligned with the task intent. Detection requires an LLM-based judge or external knowledge source.
- **Latent**: The output appears correct in isolation but causes downstream failures when consumed by other agents. Only detectable through end-to-end monitoring or formal plan verification.

The shift from explicit to latent failures corresponds to a sharp increase in detection difficulty. Most existing agent frameworks handle only explicit failures (try/catch); semantic and latent failures require fundamentally different detection machinery.

### 3.1.3 Severity: What Is the Blast Radius?

| Severity | Scope | Recovery Difficulty |
|---|---|---|
| **Local** | Affects a single agent step; downstream agents are unaware | Low — retry or repair the single step |
| **Propagating** | Corrupted output feeds into downstream agents, degrading their outputs | Medium — requires identifying the root cause agent and replaying dependent steps |
| **Systemic** | The workflow's global state is inconsistent; the original goal may be unreachable from the current state | High — requires workflow recomposition or replanning from the current state toward the goal |

## 3.2 Failure Types

We now enumerate seven failure types that collectively cover the failure space we have observed in LLM agent systems. Each is characterized along the three dimensions above, grounded in literature, and illustrated with concrete examples from multi-agent workflows.

### 3.2.1 Crash Failures

| Dimension | Value |
|---|---|
| **Origin** | Environment or Interface |
| **Manifestation** | Explicit |
| **Severity** | Local (unless unhandled, then Propagating) |
| **Detectability** | Trivial — exception handling |

**Definition.** The agent raises an unhandled exception, returns an error object, or terminates abnormally before producing output.

**Root Causes.** API key expiration, rate limiting, out-of-memory errors, malformed inputs that bypass validation, unhandled edge cases in tool-use code, dependency failures (e.g., a vector database connection drops).

**Detection.** Standard try/catch wrappers, error-field checks in response dictionaries, process health monitoring. This is the *only* failure type that most existing agent frameworks handle reliably.

**Recovery.** Retry with exponential backoff (transient errors), input simplification (complexity-induced errors), agent substitution (permanent errors). In multi-agent systems, the key challenge is determining whether the crash has corrupted shared state.

**Literature.** Crash handling is well-studied in distributed systems (Lamport, 1998; Ongaro & Ousterhout, 2014) and is the default failure model in frameworks like LangChain and AutoGen. However, treating *all* failures as crash-like — the implicit assumption of retry-based approaches — is precisely what makes current agent systems brittle against semantic failures.

### 3.2.2 Timeout Failures

| Dimension | Value |
|---|---|
| **Origin** | Environment |
| **Manifestation** | Explicit |
| **Severity** | Local to Propagating |
| **Detectability** | Easy — wall-clock monitoring |

**Definition.** The agent fails to produce output within a configured time bound. This includes both hard timeouts (the process is killed) and soft timeouts (the process completes but took unacceptably long).

**Root Causes.** LLM provider latency spikes, excessively long chain-of-thought reasoning, tool calls to slow external APIs, large context windows causing inference slowdowns, recursive self-refinement loops.

**Detection.** Wall-clock timers, async task cancellation, heartbeat monitoring for long-running agents. Soft timeouts additionally require baseline latency profiling to distinguish abnormal slowdowns from naturally variable response times.

**Recovery.** Increase timeout (if the task genuinely requires more time), simplify the input or reduce context length (if latency is input-dependent), switch to a faster model or cached response (if latency is model-dependent). A critical subtlety: a timeout may indicate that the agent is stuck in an infinite self-refinement loop (see Section 3.2.7), requiring fundamentally different recovery than a simple retry.

**Literature.** Timeout handling is standard in distributed systems. In the LLM agent context, Park et al. (2023) observe that ReAct agents frequently enter reasoning loops that only terminate via timeout, and Yao et al. (2023) note that tree-of-thought search can exhibit unbounded latency without explicit depth limits.

### 3.2.3 Hallucination

| Dimension | Value |
|---|---|
| **Origin** | Model-internal |
| **Manifestation** | Semantic |
| **Severity** | Propagating (often Systemic if undetected) |
| **Detectability** | Hard — requires external knowledge or LLM-based verification |

**Definition.** The agent produces output that is fluent and plausible but contains fabricated facts, invented references, fictitious statistics, or claims unsupported by the provided context.

This is arguably the most dangerous failure type in agentic systems because hallucinated outputs are *designed by the model's training objective to look correct*. Unlike crashes, which are self-announcing, hallucinations actively resist detection.

**Root Causes.** Training data memorization and blending, distribution shift between training and deployment domains, high sampling temperature, insufficient grounding in retrieved context, tasks that exceed the model's knowledge boundary, pressure to produce an answer when "I don't know" is not in the expected output schema.

**Subtypes.** We distinguish three hallucination subtypes following Ji et al. (2023):

1. **Intrinsic hallucination**: Output contradicts the provided input (e.g., an analysis agent reports numbers that differ from the source data).
2. **Extrinsic hallucination**: Output contains claims that cannot be verified from the input, which may or may not be true (e.g., an agent invents a citation that happens to exist).
3. **Fabrication**: Output contains entirely invented entities — fake people, organizations, studies, URLs.

**Detection.** This requires going beyond structural checks:
- **LLM-as-judge**: A separate LLM evaluates the output for factual grounding against the input context (Zheng et al., 2024). AEGIS implements this as a hallucination detection validator with configurable confidence thresholds.
- **Retrieval-based verification**: Cross-reference claims against a knowledge base or search engine (Gao et al., 2023).
- **Self-consistency**: Generate multiple responses and flag outputs that are not consistent across samples (Wang et al., 2023).
- **Formal constraint checking**: If the output maps to a structured plan (as in AEGIS's workflow recomposition), verify that all referenced actions, preconditions, and effects actually exist in the system's action space.

**Recovery.** Grounding injection (add retrieved facts to the prompt), temperature reduction (reduce sampling randomness), explicit instruction to cite sources, regeneration with a different model, or — in the neurosymbolic approach — reject the output and re-propose via formal planning.

**Literature.** Hallucination is the most extensively studied LLM failure mode. Ji et al. (2023) provide a comprehensive survey. Huang et al. (2023) study hallucination specifically in agentic tool-use settings. Min et al. (2023) demonstrate that even retrieval-augmented generation does not eliminate hallucination when the model "overrides" retrieved context with parametric knowledge.

### 3.2.4 Semantic Drift

| Dimension | Value |
|---|---|
| **Origin** | Model-internal |
| **Manifestation** | Semantic |
| **Severity** | Propagating |
| **Detectability** | Moderate — requires task-alignment evaluation |

**Definition.** The agent produces output that is factually coherent and well-formed but does not address the assigned task. The output "drifts" to a related or unrelated topic, answers a different question than the one asked, or focuses on irrelevant aspects of the input.

Semantic drift is distinct from hallucination: the output may be entirely *truthful* but *irrelevant*. A research agent asked to analyze market trends in renewable energy that instead produces a general overview of climate change has drifted — the content is real, but the task is not accomplished.

**Root Causes.** Ambiguous task specifications, long context windows that dilute the instruction signal, multi-turn conversation history that gradually shifts the agent's attention, prompt injection via user-provided content, model tendency to "satisfice" (produce any plausible-looking output) rather than precisely follow instructions.

**Detection.**
- **LLM-based alignment scoring**: Ask a judge model to rate how well the output addresses the specific task (AEGIS uses a 1-5 alignment scale with a configurable threshold).
- **Embedding similarity**: Compute cosine similarity between the task description embedding and the output embedding; flag outputs below a threshold.
- **Keyword overlap**: Simpler but surprisingly effective — check that task-specific terms appear in the output.

**Recovery.** Task clarification (rephrase the prompt with explicit constraints), few-shot examples of correct outputs, structured output formats that force task-relevant content (e.g., "your response MUST contain a section titled 'Market Trend Analysis'"), or agent substitution with a specialist agent.

**Literature.** Shi et al. (2023) demonstrate that irrelevant information in the context degrades task performance. Xu et al. (2024) show that multi-agent pipelines amplify drift — if Agent A drifts slightly, Agent B's input is already off-topic, causing compounding divergence. This "drift amplification" is a key motivation for workflow-level monitoring rather than agent-level monitoring alone.

### 3.2.5 Format and Schema Errors

| Dimension | Value |
|---|---|
| **Origin** | Interface |
| **Manifestation** | Structural |
| **Severity** | Local (if caught) to Propagating (if silently parsed) |
| **Detectability** | Easy — schema validation, type checking |

**Definition.** The agent's output violates the expected structural format — missing required JSON fields, incorrect types, broken markup, truncated responses, or extra content wrapping the expected payload (e.g., markdown code fences around JSON).

**Root Causes.** Inconsistent instruction-following across models, model updates that change output formatting behavior, prompts that insufficiently constrain the output format, token limit truncation, context window overflow causing the model to "forget" format instructions at the end of generation.

**Subtypes.**
1. **Type mismatch**: Expected a JSON object, received a string.
2. **Missing fields**: Output is valid JSON but lacks required keys.
3. **Wrapper pollution**: Output contains the correct payload but wrapped in markdown, XML, or natural language preamble.
4. **Truncation**: Output is cut off mid-token, producing invalid syntax.

**Detection.** JSON schema validation, type checking, regex-based extraction of payloads from common wrappers (AEGIS implements progressive JSON extraction: try direct parse, then strip markdown fences, then regex-extract the first JSON object). For truncation, check that outputs end with valid closing delimiters.

**Recovery.** Re-prompt with stricter format instructions, provide a concrete output example, use structured output modes (e.g., OpenAI's JSON mode, function calling), or apply a deterministic post-processing step that extracts the payload from wrapper pollution.

**Literature.** Format errors are common enough that most modern agent frameworks include some mitigation. LangChain's output parsers (Chase, 2022), Instructor (Liu, 2023), and Outlines (Willard & Louf, 2023) all address this at the framework level. However, these tools primarily handle wrapper pollution and type coercion — they do not address semantic correctness of the extracted content.

### 3.2.6 Quality Degradation

| Dimension | Value |
|---|---|
| **Origin** | Model-internal |
| **Manifestation** | Semantic (borderline Structural) |
| **Severity** | Local to Propagating |
| **Detectability** | Moderate — requires quality heuristics or LLM evaluation |

**Definition.** The agent produces output that is on-topic and correctly formatted but falls below an acceptable quality threshold — too brief, too verbose, repetitive, vague, or lacking in the depth or specificity required by the task.

Quality degradation occupies the gray zone between "failure" and "suboptimal performance." Whether it constitutes a failure depends on the application's quality requirements. In a research pipeline, a one-sentence summary of a complex topic is a quality failure; in a chatbot, it might be acceptable.

**Root Causes.** Insufficient task specification (the model doesn't know what "good" looks like), model capability limits on the specific domain, overly aggressive temperature settings (too low = generic, too high = incoherent), context window filled with irrelevant history, and the "lazy agent" pattern where the model minimizes effort.

**Detection.**
- **Length bounds**: Flag outputs that are abnormally short or long relative to the task (AEGIS uses configurable min/max output length thresholds).
- **Entropy and diversity**: Measure lexical diversity and information density; repetitive outputs have low entropy.
- **LLM-as-judge with rubric**: A judge model scores the output on task-specific quality dimensions (completeness, specificity, actionability).
- **Comparative evaluation**: Generate multiple outputs and flag those significantly worse than the median.

**Recovery.** Explicit quality requirements in the prompt ("provide at least 3 specific examples with data"), few-shot examples of high-quality outputs, temperature adjustment, or escalation to a more capable model.

**Literature.** Zheng et al. (2024) propose MT-Bench for evaluating LLM output quality. Dubois et al. (2024) show that length is a confounding factor in LLM-as-judge evaluation — longer outputs are systematically rated higher regardless of quality. This bias must be accounted for in automated quality detection to avoid rewarding verbosity over substance.

### 3.2.7 Cascading and Compositional Failures

| Dimension | Value |
|---|---|
| **Origin** | Compositional |
| **Manifestation** | Latent (initially), then Semantic or Explicit |
| **Severity** | Systemic |
| **Detectability** | Hard — requires workflow-level monitoring and state tracking |

**Definition.** A failure in one agent propagates through the workflow, causing downstream agents to fail or produce degraded outputs. The downstream failures may look like independent failures (hallucination, drift) but are actually *caused by* corrupted upstream state.

This is the failure type that most critically motivates workflow-level recovery and formal plan verification. Agent-level recovery is insufficient because the root cause agent may have already been "repaired" — the problem is that its previous corrupted output is still flowing through the pipeline.

**Root Causes.** Undetected upstream hallucination or drift that becomes input to downstream agents, partial failures where an agent succeeds on some outputs but fails on others (and downstream agents receive the mixed bag), shared state corruption (e.g., a document store is updated with hallucinated content that other agents then retrieve), and dependency chains where Agent C requires Agent B's output which requires Agent A's output — a failure at A invalidates everything downstream.

**Subtypes.**
1. **Linear cascade**: A → B → C, failure at A corrupts B and C.
2. **Fan-out cascade**: A → {B, C, D}, failure at A corrupts all parallel downstream agents.
3. **State pollution**: Agent A writes corrupted data to shared state; Agents B and C read it at different times, producing inconsistently corrupted outputs.
4. **Feedback cascade**: In iterative refinement workflows (Generate → Critique → Refine → ...), a flawed critique causes the refinement to degrade rather than improve, creating a negative feedback loop.

**Detection.** This requires *workflow-level* monitoring, not just per-agent checks:
- **Provenance tracking**: Track which agent outputs contributed to each downstream input. When a failure is detected, trace back to identify the root cause agent.
- **State consistency checking**: After each agent step, verify that the global workflow state satisfies invariants (e.g., the number of items in a list hasn't changed unexpectedly).
- **Formal plan verification**: Using planning graph analysis (as in AEGIS's GraphPlan-based recomposition), verify that the current state is still reachable from the initial state via valid actions, and that the goal is still reachable from the current state.

**Recovery.** This is where workflow-level recovery mechanisms are essential:
- **Checkpoint and replay**: Roll back to the last known-good state and re-execute from there.
- **Subgraph re-execution**: Identify the minimal set of agents that need to be re-run based on the dependency graph.
- **Workflow recomposition**: If the corrupted state makes the original plan infeasible, use formal planning to find an alternative path to the goal that avoids the failed agent or routes around the corrupted state.

**Literature.** Cascading failures are well-studied in distributed systems (Kandula et al., 2009) and microservice architectures (Zhou et al., 2021) but underexplored in LLM agent systems. Wu et al. (2023) observe cascading quality degradation in multi-agent debate but do not propose detection or recovery mechanisms. This gap — between the distributed systems community's sophisticated failure models and the LLM agent community's naive retry-based recovery — is a primary motivation for formal, workflow-level self-healing approaches.

## 3.3 Failure Type Interaction and Compounding

The seven failure types above are not independent. In practice, failures interact and compound:

| Primary Failure | Common Secondary Effect |
|---|---|
| Timeout → | Quality degradation (if partial output is used) or Crash (if hard-killed) |
| Hallucination → | Cascading failure (downstream agents consume fabricated facts) |
| Semantic drift → | Quality degradation (output is coherent but useless for the task) |
| Format error → | Crash (downstream parser fails) or Hallucination (partial parse extracts wrong content) |
| Quality degradation → | Semantic drift (if "lazy" outputs cause downstream agents to re-interpret the task) |
| Crash → | Cascading failure (if error state is passed downstream instead of caught) |

This interaction structure has a practical implication: **detection and recovery must be ordered carefully.** AEGIS's detector runs validators in a specific sequence — crash → empty → format → hallucination → semantic → quality — so that structural failures are caught before expensive LLM-based semantic checks are invoked. This is both a performance optimization and a correctness requirement: a format error may cause hallucination detection to produce a false positive if it tries to parse malformed output.

## 3.4 Comparison of Detection Approaches by Failure Type

| Failure Type | Rule-Based | Schema Validation | LLM-as-Judge | Embedding Similarity | Formal Verification |
|---|---|---|---|---|---|
| Crash | **Full** | — | — | — | — |
| Timeout | **Full** | — | — | — | — |
| Hallucination | None | None | **Primary** | Partial | Partial (structured outputs) |
| Semantic Drift | None | None | **Primary** | **Primary** | — |
| Format Error | Partial | **Full** | — | — | — |
| Quality Degradation | Partial (length) | None | **Primary** | — | — |
| Cascading | None | None | Partial | Partial | **Primary** |

**Key observation.** No single detection mechanism covers all failure types. Explicit failures (crash, timeout, format) are cheaply detectable via rules and schemas. Semantic failures (hallucination, drift, quality) require LLM-based evaluation, which adds latency and cost. Compositional failures (cascading) require workflow-level monitoring and, ideally, formal verification of plan validity. A comprehensive self-healing system must integrate all three detection paradigms.

## 3.5 Comparison of Recovery Approaches by Failure Type

| Failure Type | Retry | Prompt Repair | Agent Substitution | Workflow Recomposition | Formal Replanning |
|---|---|---|---|---|---|
| Crash | **Effective** (transient) | — | **Effective** (permanent) | — | — |
| Timeout | **Effective** (spike) | Partial (simplify) | **Effective** (slow agent) | — | — |
| Hallucination | Partial | **Effective** (grounding) | Partial | — | **Effective** (verify proposals) |
| Semantic Drift | Poor | **Effective** (clarify) | Partial | — | — |
| Format Error | **Effective** | **Effective** (examples) | — | — | — |
| Quality Degradation | Poor | **Effective** (specify) | Partial | — | — |
| Cascading | Poor | Poor | Poor | **Effective** | **Effective** |

**Key observation.** Retry — the default recovery strategy in most agent frameworks — is only effective for crash and timeout failures. For semantic failures, prompt-level repair (grounding injection, task clarification, output examples) is the primary tool. For cascading failures, neither retry nor prompt repair is sufficient; workflow-level recomposition or formal replanning is required. This motivates a *multi-level recovery architecture* that matches the recovery strategy to the failure type, escalating from cheap local repairs to expensive workflow-level recomposition only when necessary.

## 3.6 Summary

We have proposed a taxonomy of seven failure types in LLM agent systems, organized along three dimensions: origin (model-internal, interface, environment, compositional), manifestation (explicit, structural, semantic, latent), and severity (local, propagating, systemic). The taxonomy reveals two critical gaps in existing agent frameworks:

1. **Detection gap**: Most frameworks handle only explicit failures. Semantic failures (hallucination, drift, quality) and latent failures (cascading) require fundamentally different detection machinery — LLM-based evaluation and formal plan verification, respectively.

2. **Recovery gap**: Most frameworks rely on retry as the primary (often sole) recovery strategy. This is effective only for a narrow subset of failures. Semantic failures require prompt-level repair strategies, and compositional failures require workflow-level recomposition — ideally backed by formal planning guarantees to prevent recovery actions from introducing new failures.

These gaps motivate the neurosymbolic approach to self-healing: using LLMs for flexible semantic detection and creative recovery proposals, while using formal planning methods (e.g., GraphPlan) to verify that proposed recoveries are valid and goal-preserving. We explore existing systems that address portions of this space in Section 4, and the open challenges that remain in Section 7.