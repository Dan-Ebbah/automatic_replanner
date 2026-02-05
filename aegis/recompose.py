"""
AEGIS Recompose Module
======================
Dynamically restructures workflow graphs when agent-level repair fails.
This is the key novel contribution of AEGIS.
"""

import json
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
import asyncio
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from .config import AEGISConfig, FailureType, LLMProvider, RecomposeConfig
from .state import (
    FailureInfo, RepairResult, RecomposeResult, AgentInfo,
    ExecutionState, PlanningGraph, Action
)
from .registry import AgentRegistry, ToolRegistry


class AEGISRecompose:
    """
    Dynamically restructures workflow graphs to handle failures.

    Recomposition Strategies:
    - Graph planning (primary): Uses PlanningGraph to find valid alternative plans
    - LLM-based (fallback): Uses LLM to generate recomposition options

    Graph planning is preferred because it guarantees valid plans.
    """

    def __init__(
        self,
        config: AEGISConfig,
        agent_registry: AgentRegistry,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.config = config
        self.recompose_config = config.recompose
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry
        self.llm = self._create_llm()
        self.execution_state: Optional[ExecutionState] = None

    def _create_llm(self):
        """Create the LLM instance for recomposition reasoning"""
        if self.config.llm_provider == LLMProvider.OPENAI:
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=0.3,
                api_key=self.config.openai_api_key
            )
        elif self.config.llm_provider == LLMProvider.ANTHROPIC:
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=0.3,
                api_key=self.config.anthropic_api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def recompose(
        self,
        workflow_spec: Dict[str, Any],
        failure_info: FailureInfo,
        original_task: str,
        execution_history: List[Dict[str, Any]],
        repair_attempts: int
    ) -> RecomposeResult:
        """
        Generate a new workflow structure to avoid the failure.

        Uses graph planning if ToolRegistry is available, falls back to LLM.

        Args:
            workflow_spec: Current workflow specification (nodes and edges)
            failure_info: Information about the failure
            original_task: The high-level task being performed
            execution_history: History of execution steps and outputs
            repair_attempts: Number of failed repair attempts

        Returns:
            RecomposeResult with new workflow if successful
        """
        start_time = time.time()

        if self.tool_registry:
            result = self._recompose_via_planning(
                workflow_spec, failure_info, execution_history
            )
            if result.success:
                result.latency_ms = (time.time() - start_time) * 1000
                return result

        # Fall back to LLM-based recomposition
        return self._recompose_via_llm(
            workflow_spec, failure_info, original_task,
            execution_history, repair_attempts, start_time
        )

    # =========================================================================
    # Graph Planning Based Recomposition
    # =========================================================================

    def _recompose_via_planning(
        self,
        workflow_spec: Dict[str, Any],
        failure_info: FailureInfo,
        execution_history: List[Dict[str, Any]]
    ) -> RecomposeResult:
        """
        Recompose using graph planning - guaranteed valid if successful.

        This is the key insight from notebook.ipynb: use STRIPS-style planning
        to find alternative paths to the goal when an agent fails.
        """
        # Mark failed agent as unavailable
        if failure_info.agent_name:
            self.tool_registry.mark_agent_unavailable(failure_info.agent_name)

        # Compute current state from execution history
        current_state = self._compute_current_state(execution_history)
        goal = self._extract_goal(workflow_spec)

        # Get available actions (excludes failed agent)
        available_actions = self.tool_registry.get_planning_actions()

        # Build planning graph from current state
        graph = PlanningGraph(
            initial_state=current_state,
            goal=goal,
            available_actions=available_actions
        )

        if not graph.build_graph():
            return RecomposeResult(
                success=False,
                analysis="Graph planning found no valid path to goal",
                rationale="No alternative agents can achieve required capabilities"
            )

        plan_steps = graph.extract_plan()
        if plan_steps is None:
            return RecomposeResult(
                success=False,
                analysis="Goal reachable but plan extraction failed",
                rationale="Mutex conflicts prevent valid plan"
            )

        # Convert plan to workflow spec
        new_spec = self._plan_to_workflow_spec(plan_steps, workflow_spec)

        # Validate and build
        validation = self._validate_workflow(new_spec)
        if not validation["valid"]:
            return RecomposeResult(
                success=False,
                analysis="Generated workflow invalid",
                rationale=validation["reason"]
            )

        new_workflow = self._build_workflow(new_spec)
        changes = self._compute_changes(workflow_spec, new_spec)

        return RecomposeResult(
            success=True,
            new_workflow=new_workflow,
            new_workflow_spec=new_spec,
            nodes_added=changes["nodes_added"],
            nodes_removed=changes["nodes_removed"],
            edges_changed=changes["edges_changed"],
            analysis=f"Graph planning found {len(plan_steps)}-step alternative",
            rationale=f"Replaced {failure_info.agent_name} with alternative agents"
        )

    def _compute_current_state(
        self,
        execution_history: List[Dict[str, Any]]
    ) -> Set[str]:
        """
        Compute current propositions from execution history.

        Each completed action's positive_effects are added, negative_effects removed.
        This lets us replan from the current state, not from scratch.
        """
        state = set()

        for entry in execution_history:
            if entry.get("status") == "completed":
                # Add positive effects
                effects = entry.get("positive_effects", [])
                state.update(effects)

                # Remove negative effects
                neg_effects = entry.get("negative_effects", [])
                state -= set(neg_effects)

                # Also track any explicit state updates
                if "state_updates" in entry:
                    for key, value in entry["state_updates"].items():
                        if value:
                            state.add(key)
                        else:
                            state.discard(key)

        return state

    def _extract_goal(self, workflow_spec: Dict[str, Any]) -> Set[str]:
        """Extract goal propositions from workflow spec"""
        # Goal can be explicitly specified
        if "goal" in workflow_spec:
            return set(workflow_spec["goal"])

        # Or inferred from nodes that connect to END
        goal = set()
        for edge in workflow_spec.get("edges", []):
            if edge["to"] == "END":
                from_node = edge["from"]
                # Find this node and get its positive effects
                for node in workflow_spec.get("nodes", []):
                    if node["id"] == from_node:
                        goal.update(node.get("produces", []))

        return goal if goal else {"task_complete"}

    def _plan_to_workflow_spec(
        self,
        plan_steps: List[Set[Action]],
        original_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert a plan (list of parallel action sets) to workflow spec"""
        nodes = []
        edges = []

        prev_node_ids = ["START"]

        for step_idx, step in enumerate(plan_steps):
            step_node_ids = []

            for action in step:
                node_id = f"{action.name}_{step_idx}"
                nodes.append({
                    "id": node_id,
                    "agent": action.agent_id,
                    "action": action.name,
                    "config": {}
                })
                step_node_ids.append(node_id)

                # Connect from all previous nodes
                for prev_id in prev_node_ids:
                    edges.append({"from": prev_id, "to": node_id})

            prev_node_ids = step_node_ids

        # Connect final nodes to END
        for node_id in prev_node_ids:
            edges.append({"from": node_id, "to": "END"})

        return {"nodes": nodes, "edges": edges}

    # =========================================================================
    # LLM-Based Recomposition (Fallback)
    # =========================================================================

    def _recompose_via_llm(
        self,
        workflow_spec: Dict[str, Any],
        failure_info: FailureInfo,
        original_task: str,
        execution_history: List[Dict[str, Any]],
        repair_attempts: int,
        start_time: float
    ) -> RecomposeResult:
        """
        LLM-based recomposition as fallback when graph planning is unavailable
        or fails to find a solution.
        """
        # Step 1: Analyze why the current structure failed
        analysis = self._analyze_failure(
            workflow_spec, failure_info, execution_history
        )

        # Step 2: Generate recomposition options
        options = self._generate_options(
            workflow_spec, failure_info, analysis, original_task
        )

        # Step 3: Select best option
        selected_option = self._select_best_option(options, failure_info)

        # Step 4: Generate new workflow specification
        try:
            new_spec = self._generate_new_workflow(
                workflow_spec, selected_option, failure_info
            )

            # Step 5: Validate the new workflow
            validation = self._validate_workflow(new_spec)
            if not validation["valid"]:
                return RecomposeResult(
                    success=False,
                    analysis=analysis,
                    rationale=f"Generated workflow invalid: {validation['reason']}",
                    latency_ms=(time.time() - start_time) * 1000
                )

            # Step 6: Build the actual workflow
            new_workflow = self._build_workflow(new_spec)

            # Compute changes made
            changes = self._compute_changes(workflow_spec, new_spec)

            return RecomposeResult(
                success=True,
                new_workflow=new_workflow,
                new_workflow_spec=new_spec,
                nodes_added=changes["nodes_added"],
                nodes_removed=changes["nodes_removed"],
                edges_changed=changes["edges_changed"],
                analysis=analysis,
                rationale=selected_option.get("rationale", ""),
                latency_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return RecomposeResult(
                success=False,
                analysis=analysis,
                rationale=f"Recomposition failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )

    def _analyze_failure(
        self,
        workflow_spec: Dict[str, Any],
        failure_info: FailureInfo,
        execution_history: List[Dict[str, Any]]
    ) -> str:
        """Analyze why the workflow structure contributed to the failure"""

        prompt = f"""Analyze why this workflow structure led to a failure.

WORKFLOW STRUCTURE:
Nodes: {json.dumps(workflow_spec.get('nodes', []), indent=2)}
Edges: {json.dumps(workflow_spec.get('edges', []), indent=2)}

FAILURE INFORMATION:
- Failed Agent: {failure_info.agent_name}
- Failure Type: {failure_info.failure_type.value if failure_info.failure_type else 'unknown'}
- Error: {failure_info.error_message}
- Evidence: {failure_info.evidence}

EXECUTION HISTORY:
{json.dumps(execution_history[-5:], indent=2)}

Analyze:
1. What structural issues in the workflow contributed to this failure?
2. Was there missing validation or error handling?
3. Was there a problematic dependency between agents?
4. Could the workflow order have caused issues?

Provide a concise analysis in 2-3 sentences."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Analysis failed: {str(e)}"

    def _generate_options(
        self,
        workflow_spec: Dict[str, Any],
        failure_info: FailureInfo,
        analysis: str,
        original_task: str
    ) -> List[Dict[str, Any]]:
        """Generate possible recomposition options"""

        # Get available agents from registry
        available_agents = self.agent_registry.list_agents()

        prompt = f"""Based on the workflow failure analysis, generate recomposition options.

CURRENT WORKFLOW:
Nodes: {json.dumps(workflow_spec.get('nodes', []), indent=2)}
Edges: {json.dumps(workflow_spec.get('edges', []), indent=2)}

FAILURE ANALYSIS:
{analysis}

FAILED AGENT: {failure_info.agent_name}
FAILURE TYPE: {failure_info.failure_type.value if failure_info.failure_type else 'unknown'}

ORIGINAL TASK: {original_task}

AVAILABLE AGENTS TO ADD:
{json.dumps([a.to_dict() for a in available_agents], indent=2)}

Generate 2-3 recomposition options. For each option, specify:
1. What changes to make (add/remove nodes, change edges)
2. Why this would help
3. Potential risks

Respond in JSON:
{{
    "options": [
        {{
            "name": "option_name",
            "changes": {{
                "add_nodes": [{{"id": "node_id", "agent": "agent_name", "position": "before/after/parallel_to", "reference_node": "existing_node_id"}}],
                "remove_nodes": ["node_id"],
                "add_edges": [{{"from": "node_id", "to": "node_id"}}],
                "remove_edges": [{{"from": "node_id", "to": "node_id"}}]
            }},
            "rationale": "why this helps",
            "risk": "potential issues"
        }}
    ]
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)
            return result.get("options", [])
        except Exception as e:
            # Return a default safe option
            return [self._generate_default_option(failure_info)]

    def _generate_default_option(self, failure_info: FailureInfo) -> Dict[str, Any]:
        """Generate a default safe recomposition option"""

        if failure_info.failure_type == FailureType.HALLUCINATION:
            return {
                "name": "add_validator",
                "changes": {
                    "add_nodes": [{
                        "id": f"validator_after_{failure_info.agent_name}",
                        "agent": "fact_checker",
                        "position": "after",
                        "reference_node": failure_info.agent_name
                    }],
                    "remove_nodes": [],
                    "add_edges": [],
                    "remove_edges": []
                },
                "rationale": "Add fact-checking after the hallucinating agent",
                "risk": "Increases latency"
            }
        elif failure_info.failure_type == FailureType.CRASH:
            return {
                "name": "add_fallback",
                "changes": {
                    "add_nodes": [{
                        "id": f"fallback_for_{failure_info.agent_name}",
                        "agent": "generic_processor",
                        "position": "parallel_to",
                        "reference_node": failure_info.agent_name
                    }],
                    "remove_nodes": [],
                    "add_edges": [],
                    "remove_edges": []
                },
                "rationale": "Add fallback agent in parallel for redundancy",
                "risk": "Uses more resources"
            }
        else:
            return {
                "name": "add_checkpoint",
                "changes": {
                    "add_nodes": [{
                        "id": f"checkpoint_before_{failure_info.agent_name}",
                        "agent": "checkpoint",
                        "position": "before",
                        "reference_node": failure_info.agent_name
                    }],
                    "remove_nodes": [],
                    "add_edges": [],
                    "remove_edges": []
                },
                "rationale": "Add checkpoint for better rollback capability",
                "risk": "Minimal"
            }

    def _select_best_option(
        self,
        options: List[Dict[str, Any]],
        failure_info: FailureInfo
    ) -> Dict[str, Any]:
        """Select the best recomposition option"""

        if not options:
            return self._generate_default_option(failure_info)

        failure_type = failure_info.failure_type

        # Score each option
        scored_options = []
        for option in options:
            score = 0

            # Prefer options that match the failure type
            if failure_type == FailureType.HALLUCINATION:
                if "validator" in option.get("name", "").lower() or "check" in option.get("name", "").lower():
                    score += 10
            elif failure_type == FailureType.CRASH:
                if "fallback" in option.get("name", "").lower() or "redundancy" in option.get("name", "").lower():
                    score += 10
            elif failure_type == FailureType.SEMANTIC_DRIFT:
                if "clarif" in option.get("name", "").lower() or "review" in option.get("name", "").lower():
                    score += 10

            # Prefer smaller changes (lower risk)
            changes = option.get("changes", {})
            num_changes = (
                len(changes.get("add_nodes", [])) +
                len(changes.get("remove_nodes", [])) +
                len(changes.get("add_edges", [])) +
                len(changes.get("remove_edges", []))
            )
            score -= num_changes  # Penalize complex changes

            scored_options.append((score, option))

        # Sort by score and return best
        scored_options.sort(key=lambda x: x[0], reverse=True)
        return scored_options[0][1]

    def _generate_new_workflow(
        self,
        old_spec: Dict[str, Any],
        selected_option: Dict[str, Any],
        failure_info: FailureInfo
    ) -> Dict[str, Any]:
        """Generate new workflow specification by applying changes"""

        new_spec = {
            "nodes": list(old_spec.get("nodes", [])),
            "edges": list(old_spec.get("edges", []))
        }

        changes = selected_option.get("changes", {})

        # Apply node additions
        for node_to_add in changes.get("add_nodes", []):
            new_node = {
                "id": node_to_add["id"],
                "agent": node_to_add["agent"],
                "config": node_to_add.get("config", {})
            }
            new_spec["nodes"].append(new_node)

            # Add appropriate edges based on position
            position = node_to_add.get("position", "after")
            ref_node = node_to_add.get("reference_node")

            if position == "before" and ref_node:
                self._insert_node_before(new_spec, new_node["id"], ref_node)
            elif position == "after" and ref_node:
                self._insert_node_after(new_spec, new_node["id"], ref_node)
            elif position == "parallel_to" and ref_node:
                self._add_parallel_branch(new_spec, new_node["id"], ref_node)

        # Apply node removals
        for node_id in changes.get("remove_nodes", []):
            new_spec["nodes"] = [n for n in new_spec["nodes"] if n["id"] != node_id]
            new_spec["edges"] = [
                e for e in new_spec["edges"]
                if e["from"] != node_id and e["to"] != node_id
            ]

        # Apply edge additions
        for edge in changes.get("add_edges", []):
            if edge not in new_spec["edges"]:
                new_spec["edges"].append(edge)

        # Apply edge removals
        for edge in changes.get("remove_edges", []):
            new_spec["edges"] = [
                e for e in new_spec["edges"]
                if not (e["from"] == edge["from"] and e["to"] == edge["to"])
            ]

        return new_spec

    def _insert_node_before(
        self,
        spec: Dict[str, Any],
        new_node_id: str,
        reference_node_id: str
    ):
        """Insert a new node before a reference node"""

        new_edges = []
        for edge in spec["edges"]:
            if edge["to"] == reference_node_id:
                new_edges.append({"from": edge["from"], "to": new_node_id})
                new_edges.append({"from": new_node_id, "to": reference_node_id})
            else:
                new_edges.append(edge)

        spec["edges"] = new_edges

    def _insert_node_after(
        self,
        spec: Dict[str, Any],
        new_node_id: str,
        reference_node_id: str
    ):
        """Insert a new node after a reference node"""

        new_edges = []
        for edge in spec["edges"]:
            if edge["from"] == reference_node_id:
                new_edges.append({"from": reference_node_id, "to": new_node_id})
                new_edges.append({"from": new_node_id, "to": edge["to"]})
            else:
                new_edges.append(edge)

        spec["edges"] = new_edges

    def _add_parallel_branch(
        self,
        spec: Dict[str, Any],
        new_node_id: str,
        reference_node_id: str
    ):
        """Add a new node as a parallel branch to reference node"""

        incoming_edges = [e for e in spec["edges"] if e["to"] == reference_node_id]
        outgoing_edges = [e for e in spec["edges"] if e["from"] == reference_node_id]

        for edge in incoming_edges:
            spec["edges"].append({"from": edge["from"], "to": new_node_id})

        for edge in outgoing_edges:
            spec["edges"].append({"from": new_node_id, "to": edge["to"]})

    # =========================================================================
    # Shared Utilities
    # =========================================================================

    def _validate_workflow(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the workflow specification is valid"""

        nodes = spec.get("nodes", [])
        edges = spec.get("edges", [])
        node_ids = {n["id"] for n in nodes}

        # Check all edge references are valid
        for edge in edges:
            if edge["from"] not in node_ids and edge["from"] != "START":
                return {"valid": False, "reason": f"Invalid source node: {edge['from']}"}
            if edge["to"] not in node_ids and edge["to"] != "END":
                return {"valid": False, "reason": f"Invalid target node: {edge['to']}"}

        # Check workflow depth
        if len(nodes) > self.recompose_config.max_workflow_depth:
            return {"valid": False, "reason": "Workflow too deep"}

        return {"valid": True, "reason": None}

    def _build_workflow(self, spec: Dict[str, Any]) -> StateGraph:
        """Build a LangGraph StateGraph from the specification"""

        from .state import AEGISState

        graph = StateGraph(dict)

        # Add nodes
        for node in spec.get("nodes", []):
            agent_name = node.get("agent")
            agent_info = self.agent_registry.get_agent(agent_name)

            if agent_info and agent_info.agent_func:
                graph.add_node(node["id"], agent_info.agent_func)
            else:
                # Create a placeholder node
                graph.add_node(node["id"], lambda state: state)

        # Set entry point
        first_edge = next((e for e in spec.get("edges", []) if e["from"] == "START"), None)
        if first_edge:
            graph.set_entry_point(first_edge["to"])
        elif spec.get("nodes"):
            graph.set_entry_point(spec["nodes"][0]["id"])

        # Add edges
        for edge in spec.get("edges", []):
            if edge["from"] == "START":
                continue
            if edge["to"] == "END":
                graph.add_edge(edge["from"], END)
            else:
                graph.add_edge(edge["from"], edge["to"])

        return graph.compile()

    def _compute_changes(
        self,
        old_spec: Dict[str, Any],
        new_spec: Dict[str, Any]
    ) -> Dict[str, List]:
        """Compute the differences between old and new workflow specs"""

        old_node_ids = {n["id"] for n in old_spec.get("nodes", [])}
        new_node_ids = {n["id"] for n in new_spec.get("nodes", [])}

        old_edges = {(e["from"], e["to"]) for e in old_spec.get("edges", [])}
        new_edges = {(e["from"], e["to"]) for e in new_spec.get("edges", [])}

        return {
            "nodes_added": list(new_node_ids - old_node_ids),
            "nodes_removed": list(old_node_ids - new_node_ids),
            "edges_changed": [
                {"from": e[0], "to": e[1], "action": "added"}
                for e in new_edges - old_edges
            ] + [
                {"from": e[0], "to": e[1], "action": "removed"}
                for e in old_edges - new_edges
            ]
        }