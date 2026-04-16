# Aegis Repository Inventory — What's Actually There

> Generated 2026-05-06 after Claude was given access to `/Users/danieldan-ebbah/Downloads/aegis/`. Daniel's verbal description was: "aegis is just code that runs, no formal evaluation yet." This document records what's actually present so we can calibrate.

## TL;DR
There is **substantially more material** than Daniel described. This is either (a) a sign he's been working hard for a long time and underselling the work, or (b) a pile of LLM-generated artifacts he hasn't yet intellectually owned. We need to figure out which before we plan the next 12 weeks.

## Codebase

### Core framework — `aegis/` (~3,862 LOC)
| File | LOC | Purpose (per README) |
|---|---|---|
| `detector.py` | 445 | Failure detection — crashes, timeouts, hallucinations, semantic drift |
| `repair.py` | 573 | Agent repair — prompt enhancement, output regeneration, grounding injection |
| `recompose.py` | 728 | Workflow recomposition — dynamic restructuring of agent graphs |
| `wrapper.py` | 594 | LangGraph integration wrapper (drop-in pattern) |
| `state.py` | 610 | State definitions |
| `registry.py` | 269 | Agent registry |
| `config.py` | 150 | Configuration |
| `events.py` | 149 | Events |
| `mcp_adapter.py` | 126 | MCP adapter |
| `agent.py` | 106 | Agent base |

### Application layer — `app/`
- `app/chat/bot.py` — TripChatBot
- `app/api.py` — FastAPI WebSocket API
- `app/cli.py` — CLI
- `app/data/extract_data.py` — MongoDB integration
- `app/services/activity_adapter.py`

### Demos — `demos/`
- `weather_replan_demo.py` — directly relevant to thesis framing
- `trip_planner_demo.py`
- `multi_agent_demo.py`
- `agents/planner_agent.py`, `agents/weather_agent.py`

### Test multi-agent systems — `systems/`
- `research_pipeline.py` — Sequential: Research → Analyze → Summarize
- (parallel_review, iterative_refine referenced in README but not confirmed present)

### Failure injection — `injection/`
- `injector.py`

### Evaluation infrastructure — `evaluation/`
- `metrics.py`, `collector.py`, `statistics.py`

### Experiment runners — `experiments/`
- `exp_detection.py`
- `exp_repair.py`
- `exp_baselines.py`
- `exp_ablation.py`
- `exp_end_to_end.py`
- `exp_latency_cost.py`
- `exp_real_failures.py`
- `run_all.py`

### Frontend
- React + Vite + Tailwind + TS, with hooks for itinerary and websocket

### Tests — `tests/`
- `test_basic.py`
- `test_tripchatbot_healing.py`
- `test_api_llm_error_handling.py`

## Documents already drafted

### `thesis_idea.md` (33 KB, dated January 2025)
A complete-looking research proposal titled **"Self-Healing Neurosymbolic Agents: Verified Recovery Planning through LLM-Guided Diagnosis and Formal Plan Repair"**. Contains:
- 5 research questions (RQ1–RQ5)
- Architecture diagrams
- 4 baselines spec'd out
- 6 designed experiments with expected results tables
- 20+ paper literature review by phase
- 9-month roadmap (NB: not 3-month — see risk below)
- Publication targets: ICAPS, AAAI, AAMAS

### `survey/section3_taxonomy.md` (25 KB, March 2026)
Polished academic prose: "A Taxonomy of Failure Modes in LLM Agent Systems." Three-dimensional taxonomy (origin × manifestation × severity), seven concrete failure types. Reads like a publishable Section 3 / Background chapter.

### `survey/annotated_bibliography.md` (19 KB, March 2026)
Annotated bibliography. (Not yet read in detail.)

### `FIX_SUMMARY.md`
Bug fix log: MongoDB empty-plan fallback issue. Confirms the system has been actually run and debugged.

## Experiments already executed
- 3 JSON result files in `experiments/results/` (detection accuracy, repair effectiveness, dated January 2026)
- 3 JSON result files in `results/detection/` (detection accuracy, January 2026)

## Git status
- Only 4 commits total. Last commit ~April 2026 was a `.gitignore` update.
- This is unusual for ~12 months of work and suggests Daniel hasn't been committing regularly. **Risk:** loss of work, no version trail for thesis defense.

## What's notably missing or unclear
1. **A current, locked thesis statement.** `thesis_idea.md` is from Jan 2025 and may not reflect Daniel's actual direction now.
2. **Confirmation that the experiments are sound.** Existence of JSON results ≠ defensible methodology. We need to inspect.
3. **No committed thesis draft chapters.** Survey sections exist but no integrated draft.
4. **The integration with supervisor's itinerary planner.** Mentioned verbally; existence/state in code is unclear (the `app/` layer + `weather_replan_demo.py` may relate but haven't been mapped to her startup yet).
5. **A coherent story.** The codebase covers general self-healing (detector + repair + recompose); the supervisor wants the itinerary application. These are not yet aligned in writing.

## The two possible thesis framings (NEED TO RESOLVE)
1. **Original (per thesis_idea.md):** Self-Healing Neurosymbolic Agents — broad, multi-domain, 9-month plan. **Too big for 3 months.**
2. **Supervisor's framing (per recent conversation):** Integrate aegis into her itinerary planner so it adapts to weather/time disruptions; thesis is the evaluation of that integration. **Tighter, 3-month feasible.**

**These are different theses.** Daniel must pick one — and almost certainly pick #2 given the timeline.

## Honest open questions for Daniel
1. How much of the existing material did *you* write/own vs. how much was LLM-generated and never deeply reviewed?
2. Of the experiments already run — do you trust the methodology? Did you inspect the results?
3. Does `thesis_idea.md` (Jan 2025) still represent your direction, or has it evolved?
4. Are the survey/taxonomy and bibliography intellectually yours (i.e., could you defend them in a viva)?
5. What's the relationship between your `app/chat/bot.py` and your supervisor's startup planner? Is it the same system, or are they distinct?
