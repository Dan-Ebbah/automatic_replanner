# Research Proposal: Self-Healing Neurosymbolic Agents

## Verified Recovery Planning through LLM-Guided Diagnosis and Formal Plan Repair

---

## 1. Executive Summary

Large Language Model (LLM) based agents are being rapidly deployed for complex tasks like trip planning, code generation, and workflow automation. However, these agents suffer from critical reliability issues: they hallucinate invalid action sequences, fail to recover gracefully from errors, and cannot guarantee goal achievement.

This thesis proposes **Neurosymbolic Self-Healing Agents** - a hybrid architecture that combines:
- **LLM capabilities**: Semantic failure diagnosis, creative recovery proposal, natural language understanding
- **Formal planning guarantees**: Action validity verification, goal reachability proofs, mutex constraint checking

The key insight is that LLMs excel at *understanding what went wrong* and *proposing creative fixes*, while formal planners excel at *verifying those fixes are valid* and *guaranteeing goal achievement*. By combining both, we achieve self-healing agents that are both flexible and reliable.

---

## 2. Problem Statement

### 2.1 The Reliability Crisis in LLM Agents

Current LLM-based agents (AutoGPT, BabyAGI, LangChain agents) exhibit critical failure modes:

1. **Hallucinated Recovery**: When an action fails, LLMs often propose invalid recovery steps
2. **Infinite Loops**: Agents retry failed actions without meaningful adaptation
3. **Goal Divergence**: Recovery attempts may achieve something, but not the original goal
4. **Cascading Failures**: Invalid recovery actions cause additional failures

### 2.2 Limitations of Pure Formal Approaches

Classical AI planners (STRIPS, GraphPlan, PDDL) offer guarantees but suffer from:

1. **Brittleness**: Cannot handle failures not explicitly modeled
2. **Limited Creativity**: Restricted to pre-defined action sets
3. **No Semantic Understanding**: Cannot diagnose *why* something failed
4. **Domain Engineering Burden**: Requires manual specification of all possible actions

### 2.3 Research Gap

**No existing system combines LLM flexibility with formal planning guarantees for self-healing.**

| System | Flexible Recovery | Verified Plans | Semantic Diagnosis |
|--------|-------------------|----------------|-------------------|
| AutoGPT | Yes | No | Partial |
| LangChain ReAct | Yes | No | Partial |
| Classical Replanning | No | Yes | No |
| **Our Approach** | **Yes** | **Yes** | **Yes** |

---

## 3. Research Questions

### Primary Research Question
**RQ1**: How can we combine LLM-based failure diagnosis with formal plan verification to create self-healing agents that are both flexible and reliable?

### Secondary Research Questions
**RQ2**: What is the optimal division of labor between LLM reasoning and formal planning in recovery scenarios?

**RQ3**: How effectively can LLM-proposed recovery actions be translated into formally verifiable STRIPS representations?

**RQ4**: What is the trade-off between recovery flexibility and verification overhead?

**RQ5**: How does neurosymbolic self-healing compare to pure LLM and pure formal approaches across different failure types?

---

## 4. Proposed Approach

### 4.1 System Architecture: AEGIS-NS (Neurosymbolic)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AEGIS-NS ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │   MONITOR   │───►│  DIAGNOSE   │───►│   PROPOSE   │                │
│  │             │    │   (LLM)     │    │   (LLM)     │                │
│  │ Detect      │    │             │    │             │                │
│  │ failures    │    │ Classify    │    │ Generate    │                │
│  │ during      │    │ failure     │    │ recovery    │                │
│  │ execution   │    │ type &      │    │ candidates  │                │
│  │             │    │ root cause  │    │             │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│                                              │                          │
│                                              ▼                          │
│                     ┌─────────────────────────────────────┐            │
│                     │         TRANSLATE (Hybrid)          │            │
│                     │                                     │            │
│                     │  LLM proposal ──► STRIPS action     │            │
│                     │  "Use Kayak"  ──► Action(           │            │
│                     │                     name=kayak_book,│            │
│                     │                     pre={searched}, │            │
│                     │                     eff={booked})   │            │
│                     └─────────────────────────────────────┘            │
│                                              │                          │
│                                              ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    VERIFY (GraphPlan)                            │  │
│  │                                                                   │  │
│  │  For each candidate action:                                       │  │
│  │  1. Check preconditions satisfiable from current state           │  │
│  │  2. Check no mutex conflicts with required propositions          │  │
│  │  3. Build planning graph to verify goal reachability             │  │
│  │  4. Extract valid recovery plan if exists                        │  │
│  │                                                                   │  │
│  │  REJECT invalid proposals (hallucination prevention)              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                              │                          │
│                                              ▼                          │
│                     ┌─────────────────────────────────────┐            │
│                     │         SELECT & EXECUTE            │            │
│                     │                                     │            │
│                     │  Rank valid plans by:               │            │
│                     │  - Plan stability (reuse)           │            │
│                     │  - Estimated success probability    │            │
│                     │  - Resource cost                    │            │
│                     │                                     │            │
│                     │  Execute best recovery plan         │            │
│                     └─────────────────────────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Key Components

#### Component 1: Failure Monitor
- Wraps agent actions with try/catch and timeout detection
- Captures failure context (action, error, state, goal)
- Triggers healing loop on failure

#### Component 2: LLM Diagnoser
- Classifies failure type (transient, permanent, partial, resource)
- Identifies root cause using semantic understanding
- Provides structured diagnosis output

#### Component 3: LLM Recovery Proposer
- Generates multiple candidate recovery strategies
- Considers domain context and available resources
- Outputs natural language recovery descriptions

#### Component 4: Action Translator (Novel Contribution)
- Converts LLM natural language proposals to STRIPS format
- Extracts preconditions and effects
- Handles ambiguity through clarification prompts

#### Component 5: GraphPlan Verifier
- Validates translated actions are well-formed
- Checks goal reachability from current state
- Rejects invalid proposals (hallucination prevention)
- Extracts executable recovery plan

#### Component 6: Plan Executor with Monitoring
- Executes recovery plan step-by-step
- Monitors for new failures
- Recursively triggers healing if needed

### 4.3 Formal Definitions

**Definition 1 (Healing Problem)**: Given a planning problem P = (I, G, A) where execution has reached state S after partial execution, and action a ∈ A has failed, find a recovery plan π such that executing π from S achieves G.

**Definition 2 (Valid Recovery)**: A recovery plan π is valid iff:
1. All actions in π have preconditions satisfied when executed in sequence from S
2. The final state after executing π satisfies G
3. No mutex conflicts exist between concurrent actions

**Definition 3 (LLM-Proposed Action)**: An action description in natural language that must be translated to STRIPS format and verified before execution.

---

## 5. Literature Review & Papers to Read

### 5.1 Classical AI Planning (Foundation)

| Paper | Year | Key Contribution | Relevance |
|-------|------|------------------|-----------|
| **Fikes & Nilsson - STRIPS** | 1971 | Action representation with preconditions/effects | Foundation for action modeling |
| **Blum & Furst - Fast Planning Through Planning Graph Analysis** | 1997 | GraphPlan algorithm | Core verification mechanism |
| **Hoffmann & Nebel - FF Planner** | 2001 | Heuristic forward search | Baseline comparison |
| **Helmert - Fast Downward** | 2006 | State-of-the-art PDDL planner | Baseline comparison |

**Must Read First:**
```
@inproceedings{blum1997fast,
  title={Fast planning through planning graph analysis},
  author={Blum, Avrim L and Furst, Merrick L},
  booktitle={Artificial Intelligence},
  volume={90},
  pages={281--300},
  year={1997}
}
```

### 5.2 Plan Repair & Replanning

| Paper | Year | Key Contribution | Relevance |
|-------|------|------------------|-----------|
| **Fox et al. - Plan Stability** | 2006 | Metrics for measuring plan disruption | Evaluation metrics |
| **Krogt & de Weerdt - Plan Repair** | 2005 | Local repair vs. full replanning | Alternative approach comparison |
| **Nebel & Koehler - Plan Reuse** | 1995 | Reusing previous plan components | Plan stability techniques |
| **Fritz & McIlraith - Monitoring & Recovery** | 2007 | Execution monitoring in planning | Failure detection methods |

**Must Read:**
```
@inproceedings{fox2006plan,
  title={Plan stability: Replanning versus plan repair},
  author={Fox, Maria and Gerevini, Alfonso and Long, Derek and Serina, Ivan},
  booktitle={Proceedings of ICAPS},
  year={2006}
}
```

### 5.3 LLM Agents & Reasoning

| Paper | Year | Key Contribution | Relevance |
|-------|------|------------------|-----------|
| **Yao et al. - ReAct** | 2023 | Reasoning + Acting with LLMs | LLM agent baseline |
| **Shinn et al. - Reflexion** | 2023 | Self-reflection for LLM agents | LLM self-correction approach |
| **Wang et al. - Voyager** | 2023 | LLM agent with skill library | Creative action generation |
| **AutoGPT** | 2023 | Autonomous LLM agent | Industry baseline |
| **Significant Gravitas - AutoGPT Analysis** | 2023 | Failure mode analysis | Problem motivation |

**Must Read:**
```
@article{yao2022react,
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  author={Yao, Shunyu and others},
  journal={arXiv preprint arXiv:2210.03629},
  year={2022}
}

@article{shinn2023reflexion,
  title={Reflexion: Language Agents with Verbal Reinforcement Learning},
  author={Shinn, Noah and others},
  journal={arXiv preprint arXiv:2303.11366},
  year={2023}
}
```

### 5.4 Neurosymbolic AI

| Paper | Year | Key Contribution | Relevance |
|-------|------|------------------|-----------|
| **Garcez et al. - Neurosymbolic AI** | 2019 | Survey of hybrid approaches | Theoretical foundation |
| **Nye et al. - Learning Compositional Rules** | 2020 | Neural-symbolic integration | Methodology inspiration |
| **Wong et al. - LLMs for Symbolic Reasoning** | 2023 | Using LLMs for formal tasks | Direct relevance |
| **Silver et al. - PDDL Planning with LLMs** | 2023 | LLMs generating PDDL | Translation approach |
| **Liu et al. - LLM+P** | 2023 | LLM + classical planner | Most directly related work |

**Critical Reading:**
```
@article{liu2023llmp,
  title={LLM+P: Empowering Large Language Models with Optimal Planning Proficiency},
  author={Liu, Bo and others},
  journal={arXiv preprint arXiv:2304.11477},
  year={2023}
}
```

### 5.5 Self-Healing Systems

| Paper | Year | Key Contribution | Relevance |
|-------|------|------------------|-----------|
| **Kephart & Chess - Autonomic Computing** | 2003 | Self-healing systems vision | Conceptual foundation |
| **Psaier & Dustdar - Self-Healing Web Services** | 2011 | Service composition recovery | Application domain |
| **Weyns et al. - Self-Adaptive Systems** | 2013 | MAPE-K loop | Architecture pattern |

**Must Read:**
```
@article{kephart2003vision,
  title={The Vision of Autonomic Computing},
  author={Kephart, Jeffrey O and Chess, David M},
  journal={Computer},
  volume={36},
  number={1},
  pages={41--50},
  year={2003}
}
```

### 5.6 Suggested Reading Order

**Phase 1: Foundations (Week 1-2)**
1. Blum & Furst (1997) - GraphPlan
2. Kephart & Chess (2003) - Autonomic Computing
3. Fox et al. (2006) - Plan Stability

**Phase 2: LLM Agents (Week 3-4)**
4. Yao et al. (2023) - ReAct
5. Shinn et al. (2023) - Reflexion
6. Wang et al. (2023) - Voyager

**Phase 3: Neurosymbolic Integration (Week 5-6)**
7. Liu et al. (2023) - LLM+P
8. Silver et al. (2023) - PDDL with LLMs
9. Garcez et al. (2019) - Neurosymbolic AI Survey

---

## 6. Research Roadmap

### Phase 1: Foundation & Literature Review (Months 1-2)

```
Week 1-2: Deep dive into GraphPlan and plan repair literature
├── Read Blum & Furst (1997)
├── Read Fox et al. (2006)
├── Implement plan stability metrics in AEGIS
└── Deliverable: Literature review draft (planning section)

Week 3-4: LLM agents analysis
├── Read ReAct, Reflexion, Voyager papers
├── Experiment with existing LLM agent frameworks
├── Document failure modes in AutoGPT/LangChain
└── Deliverable: Literature review draft (LLM agents section)

Week 5-6: Neurosymbolic approaches
├── Read LLM+P and related work
├── Identify gaps in existing approaches
├── Refine thesis contribution statement
└── Deliverable: Complete literature review

Week 7-8: Baseline implementation
├── Implement pure LLM agent baseline
├── Implement pure replanning baseline (extend AEGIS)
├── Create failure injection framework
└── Deliverable: Baseline systems ready for comparison
```

### Phase 2: Core System Development (Months 3-5)

```
Month 3: LLM Diagnosis Component
├── Week 1: Design diagnosis prompt templates
├── Week 2: Implement failure classification system
├── Week 3: Build structured output parsing
├── Week 4: Test on synthetic failures
└── Deliverable: Working LLM diagnoser

Month 4: LLM-to-STRIPS Translator
├── Week 1: Design translation prompt templates
├── Week 2: Implement action extraction pipeline
├── Week 3: Build validation and repair mechanisms
├── Week 4: Integration with GraphPlan
└── Deliverable: Working translator with validation

Month 5: Integration & Healing Loop
├── Week 1: Integrate all components
├── Week 2: Implement plan selection heuristics
├── Week 3: Build execution monitor
├── Week 4: End-to-end testing
└── Deliverable: Complete AEGIS-NS system
```

### Phase 3: Evaluation (Months 6-7)

```
Month 6: Experiment Design & Execution
├── Week 1: Finalize experiment design
├── Week 2: Run Experiment 1 (healing success rate)
├── Week 3: Run Experiment 2 (hallucination prevention)
├── Week 4: Run Experiment 3 (plan quality)
└── Deliverable: Raw experimental results

Month 7: Analysis & Additional Experiments
├── Week 1: Statistical analysis of results
├── Week 2: Run ablation studies
├── Week 3: Run scalability experiments
├── Week 4: Compile results and visualizations
└── Deliverable: Complete evaluation chapter
```

### Phase 4: Writing & Defense (Months 8-9)

```
Month 8: Thesis Writing
├── Week 1-2: Write methodology chapter
├── Week 3-4: Write evaluation chapter
└── Deliverable: Complete thesis draft

Month 9: Revision & Defense
├── Week 1-2: Advisor feedback and revision
├── Week 3: Final polish
├── Week 4: Defense preparation
└── Deliverable: Final thesis + successful defense
```

### Visual Timeline

```
Month:    1    2    3    4    5    6    7    8    9
         ├────┼────┼────┼────┼────┼────┼────┼────┤
Phase 1: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ Literature & Baselines
Phase 2: ░░░░░░░░████████████████░░░░░░░░░░░░░░░░ System Development
Phase 3: ░░░░░░░░░░░░░░░░░░░░░░░░████████░░░░░░░░ Evaluation
Phase 4: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████ Writing & Defense

Key Milestones:
├── Month 2: Literature review complete
├── Month 5: AEGIS-NS system complete
├── Month 7: All experiments complete
└── Month 9: Thesis defense
```

---

## 7. Baselines for Comparison

### Baseline 1: Pure LLM Agent (ReAct-style)

```python
class PureLLMAgent:
    """Baseline: LLM agent with no formal verification"""

    def execute_with_recovery(self, goal: str) -> bool:
        while not self.goal_achieved(goal):
            # LLM decides next action
            action = self.llm.decide_action(self.state, goal)

            try:
                self.execute(action)
            except Exception as e:
                # LLM proposes recovery (no verification)
                recovery = self.llm.propose_recovery(e, self.state, goal)
                self.execute(recovery)  # May hallucinate!

        return self.goal_achieved(goal)
```

**Strengths**: Flexible, handles novel situations
**Weaknesses**: Hallucinations, no guarantees, may diverge from goal

### Baseline 2: Pure Formal Replanning (Your Current AEGIS)

```python
class PureFormalAgent:
    """Baseline: GraphPlan replanning with pre-defined actions only"""

    def execute_with_recovery(self, goal: Set[str]) -> bool:
        while not self.goal_achieved(goal):
            # Build plan from current state
            plan = self.graphplan(self.state, goal, self.predefined_actions)

            for action in plan:
                try:
                    self.execute(action)
                except Exception as e:
                    # Replan with remaining predefined actions
                    break  # Trigger replanning

        return self.goal_achieved(goal)
```

**Strengths**: Guaranteed valid plans, goal achievement proven
**Weaknesses**: Limited to predefined actions, can't handle novel failures

### Baseline 3: LLM + Retry (Simple Hybrid)

```python
class LLMRetryAgent:
    """Baseline: LLM agent with simple retry logic"""

    def execute_with_recovery(self, goal: str) -> bool:
        max_retries = 3

        for attempt in range(max_retries):
            try:
                plan = self.llm.generate_plan(self.state, goal)
                for action in plan:
                    self.execute(action)
                return True
            except Exception as e:
                # Just retry with error context
                self.state["last_error"] = str(e)

        return False
```

**Strengths**: Simple, some error awareness
**Weaknesses**: No real healing, just retrying

### Baseline 4: Reflexion-style Self-Correction

```python
class ReflexionAgent:
    """Baseline: LLM with self-reflection (Shinn et al., 2023)"""

    def execute_with_recovery(self, goal: str) -> bool:
        memory = []

        for episode in range(max_episodes):
            try:
                plan = self.llm.generate_plan(self.state, goal, memory)
                self.execute_plan(plan)
                return True
            except Exception as e:
                # Reflect on failure
                reflection = self.llm.reflect(e, plan, self.state)
                memory.append(reflection)

        return False
```

**Strengths**: Learns from failures within episode
**Weaknesses**: Still no formal verification, may repeat mistakes

### Comparison Matrix

| Baseline | Flexibility | Guarantees | Novel Failures | Complexity |
|----------|-------------|------------|----------------|------------|
| Pure LLM | High | None | Yes | Low |
| Pure Formal | Low | Strong | No | Medium |
| LLM + Retry | Medium | None | Partial | Low |
| Reflexion | High | None | Yes | Medium |
| **AEGIS-NS (Ours)** | **High** | **Strong** | **Yes** | **Medium** |

---

## 8. Experiments

### Experiment 1: Healing Success Rate

**Objective**: Measure percentage of failures successfully recovered across different failure types.

**Setup**:
```
Domains: 3 (Trip Planning, Workflow Orchestration, Multi-Robot)
Failure Types: 5 categories
- Transient (API timeout, temporary unavailability)
- Permanent (service deprecated, resource deleted)
- Partial (incomplete response, missing data)
- Resource (rate limited, quota exceeded)
- Novel (previously unseen error types)

Trials: 100 per failure type per domain = 1,500 total trials

Procedure:
1. Initialize agent with goal
2. Execute until failure injected
3. Measure recovery success
4. Record metrics
```

**Metrics**:
- Healing Success Rate (HSR) = Successful recoveries / Total failures
- Mean Time to Recovery (MTTR)
- Goal Achievement Rate (GAR) = Goals achieved / Total attempts

**Expected Results**:
```
                    | Pure LLM | Pure Formal | Reflexion | AEGIS-NS
─────────────────────────────────────────────────────────────────────
Transient Failures  |   60%    |    85%      |    70%    |   95%
Permanent Failures  |   40%    |    70%      |    55%    |   90%
Partial Failures    |   50%    |    60%      |    60%    |   88%
Resource Failures   |   55%    |    75%      |    65%    |   92%
Novel Failures      |   45%    |    20%      |    50%    |   85%
─────────────────────────────────────────────────────────────────────
Overall HSR         |   50%    |    62%      |    60%    |   90%
```

### Experiment 2: Hallucination Prevention

**Objective**: Measure how effectively GraphPlan catches invalid LLM proposals.

**Setup**:
```
Procedure:
1. Inject failure
2. Record all LLM-proposed recovery actions
3. Run each through GraphPlan verification
4. Categorize as: Valid, Invalid-Preconditions, Invalid-Effects, Unreachable-Goal

Trials: 500 failures across all domains
```

**Metrics**:
- Invalid Proposal Rate (IPR) = Invalid proposals / Total proposals
- Hallucination Catch Rate (HCR) = Caught by GraphPlan / Total invalid
- False Rejection Rate (FRR) = Valid proposals incorrectly rejected / Total valid

**Expected Results**:
```
LLM Proposals Analysis:
─────────────────────────
Total Proposals:           1,247
Valid Proposals:             823 (66%)
Invalid Proposals:           424 (34%)
  - Bad Preconditions:       201 (16%)
  - Bad Effects:              98 (8%)
  - Goal Unreachable:        125 (10%)

GraphPlan Verification:
─────────────────────────
Invalid Caught:           418/424 (98.6%)
False Rejections:           12/823 (1.5%)
```

### Experiment 3: Plan Quality & Stability

**Objective**: Measure quality of recovery plans compared to baselines.

**Setup**:
```
Metrics:
1. Plan Stability Score (PSS)
   - % of original plan actions preserved in recovery
   - Higher is better (less disruption)

2. Plan Optimality Ratio (POR)
   - Recovery plan length / Optimal recovery length
   - Lower is better (closer to 1.0)

3. Execution Success Rate (ESR)
   - % of recovery plans that execute successfully
   - Higher is better

Trials: 200 failures with known optimal recoveries
```

**Expected Results**:
```
                    | Pure LLM | Pure Formal | Reflexion | AEGIS-NS
─────────────────────────────────────────────────────────────────────
Plan Stability      |   32%    |    78%      |    40%    |   82%
Optimality Ratio    |   2.3    |    1.2      |    1.9    |   1.15
Execution Success   |   48%    |    95%      |    58%    |   97%
```

### Experiment 4: Ablation Study

**Objective**: Understand contribution of each component.

**Configurations**:
```
A1: Full AEGIS-NS (all components)
A2: No LLM diagnosis (random failure classification)
A3: No LLM proposals (only predefined actions)
A4: No GraphPlan verification (trust all LLM proposals)
A5: No plan stability optimization (random valid plan selection)
```

**Expected Results**:
```
Configuration | HSR   | Hallucinations | Plan Stability
──────────────────────────────────────────────────────────
A1 (Full)     | 90%   | 1.5%           | 82%
A2 (No diag)  | 75%   | 1.5%           | 70%
A3 (No prop)  | 68%   | 0%             | 85%
A4 (No verify)| 52%   | 34%            | 45%
A5 (No stab)  | 88%   | 1.5%           | 55%
```

### Experiment 5: Scalability Analysis

**Objective**: Measure performance as problem complexity increases.

**Variables**:
```
- Number of available actions: 10, 25, 50, 100, 200
- Goal complexity (number of goal propositions): 3, 5, 8, 12
- State space size: small, medium, large

Metrics:
- Recovery latency (ms)
- Memory usage (MB)
- Planning graph size
```

**Expected Results**:
```
Actions | Latency (ms) | Memory (MB) | Graph Layers
────────────────────────────────────────────────────
10      |     45       |     12      |      4
25      |     82       |     28      |      5
50      |    156       |     64      |      6
100     |    312       |    142      |      7
200     |    687       |    298      |      8
```

### Experiment 6: Real-World Case Studies

**Objective**: Demonstrate practical applicability.

**Case Study 1: Trip Planning with Service Outages**
```
Scenario: User planning trip, Expedia API goes down mid-booking
- Inject realistic API failures
- Measure recovery to alternative services (Kayak, direct airline)
- Compare user experience metrics
```

**Case Study 2: CI/CD Pipeline Recovery**
```
Scenario: Build pipeline fails due to flaky test
- Inject test failures, dependency issues, resource limits
- Measure pipeline recovery and completion rate
- Compare with current retry-based approaches
```

**Case Study 3: Multi-Agent Coordination**
```
Scenario: Robot fleet in warehouse, one robot fails
- Inject robot failures, path blockages
- Measure task reassignment and completion
- Compare coordination efficiency
```

---

## 9. Expected Contributions

### Theoretical Contributions

1. **Formal Framework for Neurosymbolic Self-Healing**
   - Definition of the healing problem in hybrid systems
   - Soundness proof: verified recoveries guarantee goal achievement
   - Completeness analysis: when can healing succeed?

2. **LLM-to-STRIPS Translation Methodology**
   - Formal specification of translation requirements
   - Error taxonomy for translation failures
   - Repair mechanisms for malformed translations

### Technical Contributions

3. **AEGIS-NS System**
   - Open-source implementation
   - Integration with LangChain/LangGraph
   - Pluggable LLM backends (Claude, GPT-4, open-source)

4. **Failure Injection Framework**
   - Systematic failure generation for evaluation
   - Reproducible benchmark suite
   - Failure taxonomy with examples

### Empirical Contributions

5. **Comprehensive Evaluation**
   - Comparison across 4 baselines
   - Multiple domains and failure types
   - Statistical significance analysis

6. **Design Guidelines**
   - When to use neurosymbolic vs. pure approaches
   - Prompt engineering best practices for diagnosis/translation
   - Trade-off analysis for practitioners

---

## 10. Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM-to-STRIPS translation too unreliable | Medium | High | Constrained output formats, few-shot examples, iterative refinement |
| GraphPlan verification too slow | Low | Medium | Incremental replanning, caching, heuristics |
| LLM API costs too high | Medium | Medium | Use open-source models for development, efficient prompting |
| Baselines perform better than expected | Low | High | Focus on novel failure types where LLM flexibility shines |
| Evaluation domains too simple | Medium | Medium | Include realistic case studies, complex goals |

---

## 11. Required Resources

### Compute
- GPU access for local LLM experiments (optional)
- Cloud API credits: ~$500 for OpenAI/Anthropic
- Standard laptop sufficient for GraphPlan component

### Software
- Python 3.10+
- LangChain/LangGraph
- Your existing AEGIS codebase
- Anthropic/OpenAI APIs

### Data
- No special datasets required
- Synthetic failure scenarios
- Public planning benchmarks (IPC domains)

### Time
- 9 months full-time OR 12-15 months part-time

---

## 12. Publication Plan

### Primary Target: Conference Paper

**Venue**: ICAPS 2025 (International Conference on Automated Planning and Scheduling)
- Submission: November 2024
- Focus: Planning + self-healing + LLM integration

**Alternative Venues**:
- AAAI 2025 (broader AI audience)
- AAMAS 2025 (multi-agent focus)
- NeurIPS 2025 (if strong empirical results)

### Secondary: Workshop Paper

**Venue**: LLM Agents Workshop @ NeurIPS/ICML
- Earlier submission possible
- Good for initial feedback

### Thesis

**Format**: Master's thesis, ~80-100 pages
**Chapters**:
1. Introduction (10 pages)
2. Background & Related Work (20 pages)
3. Methodology (20 pages)
4. Implementation (15 pages)
5. Evaluation (20 pages)
6. Discussion & Future Work (10 pages)
7. Conclusion (5 pages)

---

## 13. Future Work Extensions

After thesis completion, potential extensions include:

1. **Learning from Healing**: Use successful recoveries to improve future diagnosis
2. **Proactive Healing**: Predict failures before they occur
3. **Multi-Agent Healing**: Coordinate recovery across agent teams
4. **Human-in-the-Loop**: Interactive healing with user approval
5. **Continuous Improvement**: Fine-tune LLM on domain-specific healing patterns

---

## 14. Conclusion

This thesis proposes a novel approach to creating reliable AI agents by combining the flexibility of LLMs with the rigor of formal planning. The key insight is that self-healing requires both *understanding what went wrong* (LLM strength) and *verifying the fix is correct* (formal planning strength).

By building on your existing AEGIS system and extending it with LLM-based diagnosis and translation, you can make a meaningful contribution to one of the most pressing problems in AI today: making LLM agents reliable enough for real-world deployment.

---

## Quick Reference: Key Papers

```
MUST READ (Priority Order):
1. Blum & Furst (1997) - GraphPlan
2. Liu et al. (2023) - LLM+P
3. Yao et al. (2023) - ReAct
4. Fox et al. (2006) - Plan Stability
5. Shinn et al. (2023) - Reflexion
6. Kephart & Chess (2003) - Autonomic Computing
```

---

*Document created: January 2025*
*Last updated: January 2025*
*Author: [Your Name]*
*Advisor: [Advisor Name]*
