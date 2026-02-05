"""
Test Systems for AEGIS
======================

Multi-agent systems for testing AEGIS self-healing capabilities.

Available Systems:
- research_pipeline: Sequential Research → Analyze → Summarize
- parallel_review: Parallel reviewers → Merger (coming soon)
- iterative_refine: Cyclic Generate → Critique → Refine (coming soon)
"""

from .research_pipeline import (
    ResearchPipelineState,
    research_agent,
    analyze_agent,
    summarize_agent,
    create_research_pipeline,
    get_compiled_pipeline
)

__all__ = [
    "ResearchPipelineState",
    "research_agent",
    "analyze_agent", 
    "summarize_agent",
    "create_research_pipeline",
    "get_compiled_pipeline"
]
