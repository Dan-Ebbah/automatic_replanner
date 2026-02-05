# AEGIS: Autonomous Error-handling and Graph-recomposition for Intelligent agent Systems

A self-healing framework for LangGraph multi-agent workflows that automatically detects failures (crashes, hallucinations, semantic errors) and repairs them through agent-level fixes or dynamic workflow recomposition.

## 🎯 Key Features

- **Failure Detection**: Crashes, timeouts, hallucinations, semantic drift
- **Agent Repair**: Prompt enhancement, output regeneration, grounding injection
- **Workflow Recomposition**: Dynamic restructuring of agent graphs when repair fails
- **Drop-in Integration**: Wrap any existing LangGraph workflow

## 📁 Project Structure

```
aegis/
├── aegis/                      # Core framework
│   ├── __init__.py
│   ├── detector.py             # Failure detection module
│   ├── repair.py               # Agent repair strategies
│   ├── recompose.py            # Workflow recomposition engine
│   ├── wrapper.py              # LangGraph integration wrapper
│   ├── registry.py             # Agent registry
│   ├── state.py                # State definitions
│   └── config.py               # Configuration
│
├── systems/                    # Test multi-agent systems
│   ├── __init__.py
│   ├── research_pipeline.py    # Sequential: Research → Analyze → Summarize
│   ├── parallel_review.py      # Parallel: Multiple reviewers → Merger
│   └── iterative_refine.py     # Cyclic: Generate → Critique → Refine
│
├── injection/                  # Failure injection framework
│   ├── __init__.py
│   ├── injector.py             # Core injection logic
│   └── failures.py             # Failure type definitions
│
├── experiments/                # Experiment runners
│   ├── __init__.py
│   ├── exp_detection.py        # Detection accuracy experiments
│   ├── exp_repair.py           # Repair effectiveness experiments
│   ├── exp_recompose.py        # Recomposition quality experiments
│   └── exp_full_system.py      # End-to-end experiments
│
├── evaluation/                 # Metrics and analysis
│   ├── __init__.py
│   ├── metrics.py              # Metric calculations
│   └── collector.py            # Data collection
│
├── tests/                      # Unit tests
│   └── ...
│
├── results/                    # Experiment results (generated)
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
# Or use .env file
```

### 3. Run a Simple Test

```python
from aegis import AEGIS
from systems.research_pipeline import create_research_pipeline

# Create a standard LangGraph workflow
workflow = create_research_pipeline()

# Wrap it with AEGIS for self-healing
aegis_workflow = AEGIS.wrap(workflow)

# Run with automatic failure detection and recovery
result = aegis_workflow.invoke({"topic": "quantum computing"})
```

### 4. Run Experiments

```bash
# Test detection accuracy
python -m experiments.exp_detection

# Test repair effectiveness  
python -m experiments.exp_repair

# Test full system
python -m experiments.exp_full_system
```

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Failure Detection Rate | % of injected failures correctly detected |
| Recovery Success Rate | % of detected failures successfully healed |
| Task Completion Rate | % of tasks completed despite failures |
| Recovery Latency | Time from failure detection to recovery |
| Output Quality | Correctness of final output (vs ground truth) |

## 🔬 Research Questions

1. **RQ1**: How effectively can AEGIS detect semantic failures compared to crash-only detection?
2. **RQ2**: What is the trade-off between repair, replace, and recompose strategies?
3. **RQ3**: How does dynamic workflow recomposition affect task completion?

## 📝 Citation

```bibtex
@thesis{aegis2025,
  title={AEGIS: Self-Healing Multi-Agent Workflows through Dynamic Recomposition},
  author={Your Name},
  year={2025}
}
```

## 📄 License

MIT License
