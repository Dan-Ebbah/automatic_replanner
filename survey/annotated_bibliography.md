# Annotated Bibliography: Self-Healing in Agentic Systems

*Compiled: 2026-03-31 | For AEGIS survey paper*

---

## 1. Self-Healing & Self-Correcting LLM Agents

### 1.1 Core Self-Correction Papers

**Shinn et al. — "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023)**
- Agents reflect on failures in natural language, storing reflections in an episodic memory buffer that informs future attempts.
- Key result: Reflexion achieves 91% pass@1 on HumanEval (code gen) through iterative self-reflection.
- *Relevance*: The dominant "self-healing" baseline for LLM agents, but offers no formal guarantees — reflections can themselves be wrong, and there's no verification that the next attempt is actually better. AEGIS's GraphPlan verification addresses exactly this gap.

**Madaan et al. — "Self-Refine: Iterative Refinement with Self-Feedback" (NeurIPS 2023)**
- Single-LLM loop: generate → critique → refine, with no external feedback signal.
- Shows consistent improvement across tasks (dialogue, code, math reasoning) for 2-3 iterations, then plateaus or degrades.
- *Relevance*: Demonstrates both the promise and ceiling of pure-LLM self-correction. The degradation after ~3 iterations is a key motivation for hybrid approaches — formal verification can break the plateau by ensuring refinements are actually valid.

**Chen et al. — "Teaching Large Language Models to Self-Debug" (ICLR 2024)**
- LLMs debug their own code by examining execution results, rubber-duck explaining the code, and generating fixes.
- *Relevance*: Domain-specific self-healing (code), but the diagnose-then-fix pattern generalizes. The key limitation is that self-debugging is only as good as the LLM's understanding of the failure — it can't catch failures it doesn't recognize.

**Pan et al. — "Automatically Correcting Large Language Models: Surveying the Landscape of Diverse Self-Correction Strategies" (TACL 2024)**
- Comprehensive survey of self-correction: intrinsic (self-feedback), extrinsic (tool/verifier feedback), and hybrid strategies.
- Key finding: intrinsic self-correction alone is unreliable — models often "correct" right answers into wrong ones. External verification signals dramatically improve correction.
- *Relevance*: Directly supports the AEGIS thesis — LLMs need external verification (like GraphPlan) to self-correct reliably. This paper is essential for your Related Work section.

**Huang et al. — "Large Language Models Cannot Self-Correct Reasoning Yet" (ICLR 2024)**
- Provocative result: without external feedback, LLMs' self-correction on reasoning tasks is no better than random — they flip correct answers to incorrect at roughly the same rate as vice versa.
- *Relevance*: Critical motivating result for your work. Pure LLM self-healing is fundamentally limited. The formal planning component in AEGIS is not optional — it's necessary.

**Gou et al. — "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing" (ICLR 2024)**
- LLMs use external tools (search engines, code interpreters, calculators) to verify and correct their own outputs.
- *Relevance*: Shows that tool-augmented self-correction works much better than intrinsic self-correction. AEGIS's GraphPlan verifier is, in this framing, a "tool" that provides formal verification feedback to the LLM's recovery proposals.

### 1.2 Agent Recovery & Resilience

**Yao et al. — "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)**
- Interleaved reasoning traces and actions, enabling agents to reason about failures and adjust.
- *Relevance*: Foundation for all modern LLM agents. However, ReAct's error handling is implicit (just reason harder) — there's no explicit failure detection or recovery mechanism.

**Zhou et al. — "Language Agent Tree Search (LATS)" (NeurIPS 2023 Workshop, ICML 2024)**
- Combines LLM agents with Monte Carlo Tree Search for exploration and backtracking.
- When an action fails, LATS can backtrack to a previous state and explore alternatives.
- *Relevance*: A search-based approach to recovery — complementary to AEGIS's planning-based approach. LATS explores the space empirically; AEGIS verifies recovery plans formally. Interesting comparison baseline.

**Wang et al. — "Voyager: An Open-Ended Embodied Agent with Large Language Models" (NeurIPS 2023 Spotlight)**
- LLM agent that writes and stores reusable skills (code), with a self-verification module.
- When a skill fails, Voyager uses environment feedback + self-reflection to debug and retry.
- *Relevance*: Demonstrates LLM-based self-healing in embodied agents. The skill library is a form of "learning from healing" — successful recoveries become reusable. Relevant to your Future Work section.

**Renze & Guven — "Self-Reflection in LLM Agents: Effects on Problem-Solving Performance" (arXiv 2024)**
- Systematic study of when self-reflection helps vs. hurts across different tasks and models.
- Key finding: self-reflection helps most when external feedback is available and the task has verifiable answers.
- *Relevance*: Supports the case for verification-augmented self-healing. Planning tasks are exactly the kind of "verifiable" domain where AEGIS's approach should shine.

**Kim et al. — "Language Models as Agent Models" (EMNLP 2023)**
- Formalizes what it means for an LLM to model another agent's behavior, including predicting failures.
- *Relevance*: Theoretical foundation for LLM-based failure diagnosis — understanding *why* an agent failed requires modeling the agent's decision process.

---

## 2. Neurosymbolic Planning with LLMs

### 2.1 LLM + Classical Planner Hybrids

**Liu et al. — "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency" (arXiv 2023)**
- LLM translates natural language problems into PDDL, then calls an off-the-shelf classical planner (Fast Downward).
- Key result: dramatically outperforms LLMs alone on planning benchmarks.
- *Relevance*: Most directly related to AEGIS. LLM+P does LLM→PDDL→planner for *initial* planning; AEGIS does this for *recovery* planning. Key difference: AEGIS must handle partial states and failed actions, not just clean initial states.

**Silver et al. — "Generalized Planning in PDDL Domains with Pretrained Large Language Models" (AAAI 2024)**
- LLMs generate PDDL domain models and problem instances from natural language, evaluated on IPC benchmarks.
- Key finding: LLMs generate syntactically correct PDDL ~60-70% of the time, but semantic correctness is lower.
- *Relevance*: Directly relevant to AEGIS's "LLM-to-STRIPS translator" component. The ~30-40% error rate in PDDL generation is exactly why formal verification is needed.

**Guan et al. — "Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning" (NeurIPS 2023)**
- LLMs construct PDDL world models, then use classical planners for task planning. Includes an iterative correction loop where planner errors are fed back to the LLM.
- *Relevance*: The correction loop (planner rejects → LLM fixes) is closely related to AEGIS's verify-then-repropose pattern. Good comparison point.

**Valmeekam et al. — "On the Planning Abilities of Large Language Models" (NeurIPS 2023)**
- Systematic evaluation showing LLMs are poor autonomous planners — GPT-4 solves only ~12% of Blocksworld problems correctly.
- *Relevance*: Key motivating result. LLMs can't plan reliably alone, which is why the formal planning component is essential. Cite this to justify the hybrid approach.

**Valmeekam et al. — "PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change" (NeurIPS 2023)**
- Benchmark for evaluating LLM planning across multiple domains (Blocksworld, Logistics, etc.).
- *Relevance*: Potential evaluation benchmark for AEGIS. Could test recovery planning on standard planning domains.

**Kambhampati et al. — "LLMs Can't Plan, But Can Help Planning" (arXiv 2024)**
- Position paper arguing LLMs should be used as idea generators for planners, not as planners themselves.
- Proposes LLM-Modulo framework: LLM generates candidates, external verifiers check them.
- *Relevance*: This is essentially the theoretical justification for AEGIS's architecture. Kambhampati's framing of "LLM as generator, verifier as filter" maps directly to your LLM proposer + GraphPlan verifier.

**Xie et al. — "Translating Natural Language to Planning Goals with Large Language Models" (arXiv 2023)**
- Focuses specifically on the NL→PDDL goal translation step, using few-shot prompting.
- *Relevance*: Relevant to the translation component of AEGIS. Goal translation is arguably easier than full action translation, so this represents a lower bound on what's achievable.

### 2.2 Formal Verification of LLM Outputs

**Olausson et al. — "Is Self-Repair a Silver Bullet for Code Generation?" (ICLR 2024)**
- Studies when LLM self-repair works for code, finding it helps weak models more than strong ones, and that repair quality depends heavily on error message informativeness.
- *Relevance*: The error message = diagnostic information principle. AEGIS's LLM diagnoser produces structured diagnostics, which is analogous to providing informative "error messages" for the recovery proposer.

**Stechly et al. — "GPT-4 Doesn't Know It's Wrong: An Analysis of Iterative Prompting for Reasoning Problems" (arXiv 2024)**
- Shows that LLMs can't reliably detect their own planning errors even with iterative prompting.
- *Relevance*: Another nail in the coffin for pure-LLM self-correction in planning. External verification is necessary.

---

## 3. Multi-Agent Failure & Recovery

### 3.1 Multi-Agent LLM Systems

**Wu et al. — "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (arXiv 2023, COLM 2024)**
- Framework for multi-agent LLM systems with conversation-based coordination.
- Error handling: basic retry and human-in-the-loop escalation. No formal recovery mechanisms.
- *Relevance*: Industry-standard multi-agent framework, but illustrates the recovery gap — even sophisticated frameworks default to retry-based recovery.

**Hong et al. — "MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework" (ICLR 2024)**
- Multi-agent framework using Standardized Operating Procedures (SOPs) to structure agent collaboration.
- SOPs provide implicit failure prevention via structured handoffs, but no explicit recovery.
- *Relevance*: Shows that structured coordination reduces failures but doesn't eliminate them. When SOPs fail, there's no recovery mechanism — this is the gap AEGIS fills.

**Li et al. — "CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society" (NeurIPS 2023)**
- Role-playing multi-agent framework for task completion.
- Observes "flipping" and "degeneration" failure modes where agents converge to repetitive or off-topic dialogue.
- *Relevance*: Documents compositional failure modes in multi-agent systems that individual agent monitoring can't catch.

**Chen et al. — "AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors" (ICLR 2024)**
- Dynamic agent group formation and role assignment for collaborative tasks.
- *Relevance*: Agent substitution as a recovery strategy — when an agent fails, the group can reconfigure. Related to AEGIS's workflow recomposition concept.

### 3.2 Self-Adaptive Systems (Classical)

**Kephart & Chess — "The Vision of Autonomic Computing" (IEEE Computer 2003)**
- Seminal paper defining self-* properties: self-configuration, self-optimization, self-healing, self-protection.
- *Relevance*: Foundational framing. AEGIS implements the self-healing component of autonomic computing for LLM agent systems.

**Weyns et al. — "On Patterns for Decentralized Control in Self-Adaptive Systems" (Journal of Software Engineering for Self-Adaptive Systems, 2013)**
- Surveys MAPE-K (Monitor-Analyze-Plan-Execute over shared Knowledge) and decentralized variants.
- *Relevance*: AEGIS's healing loop (Monitor→Diagnose→Propose→Verify→Execute) is a refinement of MAPE-K for the LLM agent context. The "Verify" step is AEGIS's novel addition.

**Ghahremani et al. — "Towards Self-Healing in Microservice Architectures" (ACM SEAMS 2020)**
- Self-healing patterns for microservices: circuit breakers, bulkheads, fallback chains.
- *Relevance*: Engineering patterns that can be adapted for agent systems. AEGIS's agent substitution and workflow recomposition are analogous to microservice fallback and circuit breaker patterns.

---

## 4. Agent Benchmarks & Failure Analysis

### 4.1 Agent Benchmarks

**Liu et al. — "AgentBench: Evaluating LLMs as Agents" (ICLR 2024)**
- Benchmark across 8 environments (web, code, games, DB, etc.) testing LLM agent capabilities.
- Key finding: even GPT-4 succeeds on only ~14% of hard tasks. Open models fare much worse.
- *Relevance*: Quantifies the reliability problem. 86% failure rate on hard tasks = massive opportunity for self-healing.

**Jimenez et al. — "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (ICLR 2024)**
- Benchmark of real GitHub issues requiring agents to navigate repos, understand codebases, write patches.
- *Relevance*: Real-world agent failure benchmark. Recovery from wrong patches, failed tests, etc. is a self-healing problem.

**Qin et al. — "ToolBench / ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs" (ICLR 2024)**
- Large-scale benchmark for tool-use with real APIs.
- Documents failure modes: wrong API selection, parameter hallucination, multi-step tool chains failing.
- *Relevance*: Tool-use failures are a major category in your taxonomy (Interface failures, Section 3.2.5). This benchmark could be used to evaluate AEGIS on tool-use recovery.

**Zhou et al. — "WebArena: A Realistic Web Environment for Building Autonomous Agents" (ICLR 2024)**
- Realistic web benchmark requiring multi-step navigation, form filling, information retrieval.
- Current best agents solve ~14-35% of tasks.
- *Relevance*: Another benchmark showing agent unreliability. Web tasks frequently fail at intermediate steps — recovery is essential.

**Yao et al. — "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains" (arXiv 2024)**
- Focuses on multi-turn agent interactions where recovery from misunderstandings and errors is part of the task.
- *Relevance*: Explicitly benchmarks recovery behavior, not just task completion.

### 4.2 Failure Analysis Studies

**Huang et al. — "Understanding the Planning of LLM Agents: A Survey" (arXiv 2024)**
- Survey specifically on LLM agent planning, covering generation, verification, and recovery.
- *Relevance*: Directly overlapping survey — important to cite and differentiate from. Your survey focuses specifically on *self-healing*, which this survey treats as one subsection.

**Ruan et al. — "Identifying the Risks of LM Agents with an LM-Emulated Sandbox" (ICLR 2024)**
- Uses LLM-emulated environments to identify failure modes and safety risks of LLM agents.
- *Relevance*: Failure mode identification methodology. AEGIS's failure injection framework is conceptually similar.

**Xi et al. — "The Rise and Potential of Large Language Model Based Agents: A Survey" (arXiv 2023)**
- Comprehensive survey of LLM agents covering perception, brain, and action modules.
- *Relevance*: Broad survey that provides context. Self-healing is not a focus — this is the gap your survey fills.

**Wang et al. — "A Survey on Large Language Model Based Autonomous Agents" (Frontiers of CS 2024)**
- Another broad LLM agent survey, with sections on planning, memory, and tool use.
- *Relevance*: Similar to Xi et al. — good for context, but self-healing is underexplored.

---

## 5. Classical Planning & Plan Repair (Foundations)

**Blum & Furst — "Fast Planning Through Planning Graph Analysis" (AI Journal 1997)**
- Introduces GraphPlan: planning graph construction + backward search for solution extraction.
- *Relevance*: Core algorithm underlying AEGIS's verification component.

**Fox et al. — "Plan Stability: Replanning versus Plan Repair" (ICAPS 2006)**
- Defines plan stability metrics and shows local repair preserves more of the original plan than full replanning.
- *Relevance*: AEGIS should prefer plan repair over replanning to minimize disruption — this paper provides the theoretical justification and metrics.

**van der Krogt & de Weerdt — "Plan Repair as an Extension of Planning" (ICAPS 2005)**
- Formalizes plan repair as a constrained planning problem that reuses portions of the failed plan.
- *Relevance*: AEGIS's recovery planning can be framed as plan repair — this connects your work to the classical planning literature.

**Fritz & McIlraith — "Monitoring Plan Optimality during Execution" (ICAPS 2007)**
- Runtime monitoring of plan execution to detect when the current plan becomes suboptimal due to environmental changes.
- *Relevance*: Proactive healing — detecting that a plan should be revised before it fails. Relevant to AEGIS's monitor component.

**Nebel & Koehler — "Plan Reuse versus Plan Generation: A Theoretical and Empirical Analysis" (AI Journal 1995)**
- Shows plan reuse is theoretically as hard as plan generation in the worst case, but empirically much faster.
- *Relevance*: Justifies AEGIS's approach of reusing plan components during recovery.

---

## 6. Additional Important References

**Ji et al. — "Survey of Hallucination in Natural Language Generation" (ACM Computing Surveys 2023)**
- Definitive hallucination survey covering causes, detection, and mitigation.
- *Relevance*: Foundation for your hallucination failure type (Section 3.2.3).

**Zheng et al. — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023)**
- Establishes LLM-as-judge methodology for evaluating LLM outputs.
- *Relevance*: Foundation for using LLMs as failure detectors in AEGIS.

**Lamport — "The Part-Time Parliament (Paxos)" (ACM TOCS 1998)**
- Foundational consensus protocol for fault-tolerant distributed systems.
- *Relevance*: Theoretical foundation for reasoning about failures in distributed agent systems.

**Ongaro & Ousterhout — "In Search of an Understandable Consensus Algorithm (Raft)" (USENIX ATC 2014)**
- Understandable consensus protocol for distributed systems.
- *Relevance*: Agent coordination and recovery can draw on distributed systems failure models.

---

## Summary Statistics

| Category | Papers | Key Gap Identified |
|---|---|---|
| Self-healing/correction | 11 | Pure LLM correction unreliable without external verification |
| Neurosymbolic planning | 9 | LLMs can't plan alone; hybrid systems work but no one does this for *recovery* |
| Multi-agent failure | 7 | Frameworks lack formal recovery; compositional failures understudied |
| Benchmarks & analysis | 9 | Agents fail 65-86% on hard tasks; recovery barely measured |
| Classical planning | 5 | Plan repair theory exists but hasn't been applied to LLM agent recovery |
| Supporting references | 4 | — |
| **Total** | **45** | — |

## Key Takeaway for Survey Structure

The literature reveals a clear gap at the intersection of three well-studied areas:
1. **LLM self-correction** (active area, but shown to be unreliable without external verification)
2. **LLM + formal planning** (growing area, but focused on initial planning, not recovery)
3. **Self-healing systems** (mature in distributed systems, barely explored for LLM agents)

**Nobody is doing verified self-healing for LLM agents.** This is your thesis contribution.