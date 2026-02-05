"""
Research Pipeline System
========================
A simple sequential multi-agent workflow for testing AEGIS.

Pipeline: Research → Analyze → Summarize

This is a good test system because:
- Simple linear structure
- Clear task semantics
- Easy to inject failures
- Representative of real use cases
"""

from typing import TypedDict, Optional, List, Any
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# State Definition
# ============================================================================

class ResearchPipelineState(TypedDict, total=False):
    """State for the research pipeline"""
    
    # Input
    topic: str
    
    # Agent outputs
    research_data: str
    analysis: str
    summary: str
    
    # Metadata
    errors: List[str]
    execution_log: List[dict]


# ============================================================================
# Agent Functions
# ============================================================================

def get_llm():
    """Get LLM instance"""
    return ChatOpenAI(
        model="gpt-4o-mini",  # Using mini for cost efficiency in testing
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )


def research_agent(state: ResearchPipelineState) -> dict:
    """
    Agent 1: Research Agent
    Gathers information about the given topic.
    """
    topic = state.get("topic", "")
    
    llm = get_llm()
    
    prompt = f"""You are a research agent. Gather comprehensive information about the following topic.

TOPIC: {topic}

Provide:
1. Key facts and background
2. Recent developments
3. Important statistics or data
4. Key players or stakeholders
5. Relevant sources or references

Be thorough and factual. Only include verified information."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "research_data": response.content,
        "execution_log": state.get("execution_log", []) + [{
            "agent": "research",
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }]
    }


def analyze_agent(state: ResearchPipelineState) -> dict:
    """
    Agent 2: Analysis Agent
    Analyzes the research data to extract insights.
    """
    research_data = state.get("research_data", "")
    topic = state.get("topic", "")
    
    if not research_data:
        return {
            "analysis": "",
            "errors": state.get("errors", []) + ["No research data to analyze"]
        }
    
    llm = get_llm()
    
    prompt = f"""You are an analysis agent. Analyze the following research data about "{topic}".

RESEARCH DATA:
{research_data}

Provide:
1. Key insights and patterns
2. Strengths and weaknesses
3. Opportunities and threats
4. Recommendations
5. Areas needing further investigation

Be analytical and insightful."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "analysis": response.content,
        "execution_log": state.get("execution_log", []) + [{
            "agent": "analyze",
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }]
    }


def summarize_agent(state: ResearchPipelineState) -> dict:
    """
    Agent 3: Summarization Agent
    Creates a concise summary of the research and analysis.
    """
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")
    topic = state.get("topic", "")
    
    llm = get_llm()
    
    prompt = f"""You are a summarization agent. Create a concise executive summary about "{topic}".

RESEARCH DATA:
{research_data[:2000]}

ANALYSIS:
{analysis[:2000]}

Create a summary that:
1. Captures the key points (3-5 bullet points)
2. Highlights the most important insights
3. Provides actionable recommendations
4. Is concise but comprehensive (300-500 words)"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "summary": response.content,
        "execution_log": state.get("execution_log", []) + [{
            "agent": "summarize",
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }]
    }


# ============================================================================
# Workflow Definition
# ============================================================================

def create_research_pipeline() -> StateGraph:
    """
    Create the research pipeline workflow.
    
    Returns:
        Compiled LangGraph workflow
    """
    
    # Create the graph
    workflow = StateGraph(ResearchPipelineState)
    
    # Add nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("analyze", analyze_agent)
    workflow.add_node("summarize", summarize_agent)
    
    # Add edges (sequential flow)
    workflow.set_entry_point("research")
    workflow.add_edge("research", "analyze")
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", END)
    
    return workflow


def get_compiled_pipeline():
    """Get a compiled version of the pipeline"""
    return create_research_pipeline().compile()


# ============================================================================
# Test Function
# ============================================================================

def test_research_pipeline():
    """Test the research pipeline with a sample topic"""
    
    pipeline = get_compiled_pipeline()
    
    result = pipeline.invoke({
        "topic": "The impact of artificial intelligence on healthcare",
        "errors": [],
        "execution_log": []
    })
    
    print("=" * 60)
    print("RESEARCH PIPELINE TEST")
    print("=" * 60)
    print(f"\nTopic: {result.get('topic')}")
    print(f"\nResearch Data:\n{result.get('research_data', '')[:500]}...")
    print(f"\nAnalysis:\n{result.get('analysis', '')[:500]}...")
    print(f"\nSummary:\n{result.get('summary', '')[:500]}...")
    print(f"\nExecution Log: {result.get('execution_log')}")
    
    return result


if __name__ == "__main__":
    test_research_pipeline()
