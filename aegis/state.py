"""
AEGIS State Definitions
=======================
Defines state classes and data structures used throughout AEGIS.
"""

from dataclasses import dataclass, field
from typing import TypedDict, Optional, List, Dict, Any, Annotated, FrozenSet, Callable, Set
from datetime import datetime
from enum import Enum, auto
import operator

from .config import FailureType, RecoveryStrategy

class ActionStatus(Enum):
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass(frozen=True)
class Action:
    name: str
    agent_id: str
    preconditions: FrozenSet[str]
    positive_effects: FrozenSet[str]
    negative_effects: FrozenSet[str]
    capability: Optional[str] = None

    def _is_applicable(self, state: Set[str]) -> bool:
        return self.preconditions.issubset(state)

    def apply(self, state: Set[str]) -> Set[str]:
        return (state - self.negative_effects).union(self.positive_effects)

@dataclass
class AgentTool:
    name: str
    agent_id: str
    description: str
    pre_conditions: FrozenSet[str]
    negative_effects: FrozenSet[str]
    positive_effects: FrozenSet[str] = frozenset()
    capability: str | None = None

    parameters_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    execute_fn: Callable[..., Any] | None = None

    def to_planning_action(self) -> Action:
        return Action(
            name=self.name,
            agent_id=self.agent_id,
            preconditions=self.pre_conditions,
            positive_effects=self.positive_effects,
            negative_effects=self.negative_effects,
            capability=self.capability
        )

    def to_llm_tool_spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters_schema.get("properties", {}),
                "required": self.parameters_schema.get("required", [])
            }
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        if self.execute_fn is None:
            raise RuntimeError(f"Tool {self.name} has no execution function defined")

        result = await self.execute_fn(**kwargs)
        return result


# ============================================================================
# Execution State for Graph Planning
# ============================================================================

@dataclass
class ExecutionState:
    """Tracks the current state of workflow execution for replanning"""
    current_propositions: Set[str] = field(default_factory=set)
    action_statuses: Dict[str, ActionStatus] = field(default_factory=dict)
    goal: Set[str] = field(default_factory=set)

    def get_completed_actions(self) -> List[str]:
        return [
            name for name, status in self.action_statuses.items()
            if status == ActionStatus.COMPLETED
        ]

    def get_pending_actions(self) -> List[str]:
        return [
            name for name, status in self.action_statuses.items()
            if status in (ActionStatus.PENDING, ActionStatus.EXECUTING)
        ]


# ============================================================================
# Planning Graph for Recomposition
# ============================================================================

@dataclass
class PlanningGraph:
    """GraphPlan-based planner for computing valid action sequences"""

    @dataclass
    class Layer:
        propositions: Set[str] = field(default_factory=set)
        actions: Set[Action] = field(default_factory=set)
        proposition_mutexes: Set[FrozenSet[str]] = field(default_factory=set)
        action_mutex: Set[FrozenSet[Action]] = field(default_factory=set)

    initial_state: Set[str]
    goal: Set[str]
    available_actions: List[Action]
    layers: List[Layer] = field(default_factory=list)

    def build_graph(self, max_layers: int = 20) -> bool:
        """Build planning graph. Returns True if goal is reachable."""
        initial_layer = self.Layer(propositions=self.initial_state.copy())
        self.layers = [initial_layer]

        for _ in range(max_layers):
            current_props = self.layers[-1].propositions

            if self.goal.issubset(current_props):
                if not self._goal_has_mutex(self.layers[-1]):
                    return True

            next_actions = set()
            for action in self.available_actions:
                if action._is_applicable(current_props):
                    if not self._preconditions_mutex(action, self.layers[-1]):
                        next_actions.add(action)

            next_props = current_props.copy()
            for action in next_actions:
                next_props |= action.positive_effects

            action_mutexes = self._compute_action_mutexes(next_actions)
            prop_mutexes = self._compute_proposition_mutexes(
                next_props, next_actions, action_mutexes, self.layers[-1]
            )

            new_layer = self.Layer(
                propositions=next_props,
                actions=next_actions,
                proposition_mutexes=prop_mutexes,
                action_mutex=action_mutexes
            )

            # Fixed point - no more progress possible
            if (next_props == current_props and
                prop_mutexes == self.layers[-1].proposition_mutexes):
                return False

            self.layers.append(new_layer)

        return False

    def _goal_has_mutex(self, layer: Layer) -> bool:
        goal_list = list(self.goal)
        for i, p1 in enumerate(goal_list):
            for p2 in goal_list[i + 1:]:
                if frozenset({p1, p2}) in layer.proposition_mutexes:
                    return True
        return False

    def _preconditions_mutex(self, action: Action, layer: Layer) -> bool:
        preconds = list(action.preconditions)
        for i, p1 in enumerate(preconds):
            for p2 in preconds[i + 1:]:
                if frozenset({p1, p2}) in layer.proposition_mutexes:
                    return True
        return False

    def _compute_action_mutexes(self, actions: Set[Action]) -> Set[FrozenSet[Action]]:
        mutexes = set()
        action_list = list(actions)
        for i, a1 in enumerate(action_list):
            for a2 in action_list[i + 1:]:
                # Inconsistent effects
                if (a1.positive_effects & a2.negative_effects or
                    a2.positive_effects & a1.negative_effects):
                    mutexes.add(frozenset({a1, a2}))
                    continue
                # Interference
                if (a1.negative_effects & a2.preconditions or
                    a2.negative_effects & a1.preconditions):
                    mutexes.add(frozenset({a1, a2}))
        return mutexes

    def _compute_proposition_mutexes(
        self,
        propositions: Set[str],
        actions: Set[Action],
        action_mutexes: Set[FrozenSet[Action]],
        previous_layer: Layer
    ) -> Set[FrozenSet[str]]:
        mutexes = set()
        prop_list = list(propositions)

        # Build achievers map: which actions produce each proposition
        achievers: Dict[str, Set[Action]] = {p: set() for p in propositions}
        for action in actions:
            for prop in action.positive_effects:
                if prop in achievers:
                    achievers[prop].add(action)

        for i, p1 in enumerate(prop_list):
            for p2 in prop_list[i + 1:]:
                ways_to_achieve_p1 = achievers.get(p1, set())
                ways_to_achieve_p2 = achievers.get(p2, set())

                p1_can_persist = p1 in previous_layer.propositions
                p2_can_persist = p2 in previous_layer.propositions
                persist_mutex = frozenset({p1, p2}) in previous_layer.proposition_mutexes

                # If both can persist and weren't mutex before, they're not mutex now
                if p1_can_persist and p2_can_persist and not persist_mutex:
                    continue

                # If one can persist (non-mutex) and the other has any achiever,
                # they can coexist (persist + action that doesn't delete p1)
                if p1_can_persist and not persist_mutex:
                    # p1 persists, check if any achiever of p2 deletes p1
                    p1_safe = False
                    for a2 in ways_to_achieve_p2:
                        if p1 not in a2.negative_effects:
                            p1_safe = True
                            break
                    if p1_safe or not ways_to_achieve_p2:
                        continue

                if p2_can_persist and not persist_mutex:
                    # p2 persists, check if any achiever of p1 deletes p2
                    p2_safe = False
                    for a1 in ways_to_achieve_p1:
                        if p2 not in a1.negative_effects:
                            p2_safe = True
                            break
                    if p2_safe or not ways_to_achieve_p1:
                        continue

                # Check if all pairs of achievers are mutex
                all_pairs_mutex = True

                # If either has no achievers and can't persist, they're unreachable
                if not ways_to_achieve_p1 and not p1_can_persist:
                    continue
                if not ways_to_achieve_p2 and not p2_can_persist:
                    continue

                # Check action pairs
                for a1 in ways_to_achieve_p1:
                    for a2 in ways_to_achieve_p2:
                        if a1 == a2:
                            # Same action achieves both - not mutex
                            all_pairs_mutex = False
                            break
                        if frozenset({a1, a2}) not in action_mutexes:
                            # Non-mutex action pair exists
                            all_pairs_mutex = False
                            break
                    if not all_pairs_mutex:
                        break

                if all_pairs_mutex and ways_to_achieve_p1 and ways_to_achieve_p2:
                    mutexes.add(frozenset({p1, p2}))

        return mutexes

    def extract_plan(self) -> Optional[List[Set[Action]]]:
        """Extract a plan via backward search. Returns list of parallel action sets."""
        if not self.layers or not self.goal.issubset(self.layers[-1].propositions):
            return None
        return self._backward_search(len(self.layers) - 1, self.goal)

    def _backward_search(
        self, layer_index: int, subgoal: Set[str]
    ) -> Optional[List[Set[Action]]]:
        if layer_index == 0:
            return [] if subgoal.issubset(self.layers[0].propositions) else None

        layer = self.layers[layer_index]

        goal_achievers: Dict[str, List[Optional[Action]]] = {}
        for goal in subgoal:
            achievers = []
            for action in layer.actions:
                if goal in action.positive_effects:
                    achievers.append(action)
            if goal in self.layers[layer_index - 1].propositions:
                achievers.append(None)  # Can persist from previous layer
            goal_achievers[goal] = achievers

        selected = self._select_achievers(goal_achievers, layer)
        if selected is None:
            return None

        new_goals = set()
        for action in selected:
            if action is not None:
                new_goals |= action.preconditions

        for goal, achiever in zip(subgoal, selected):
            if achiever is None:
                new_goals.add(goal)

        sub_plan = self._backward_search(layer_index - 1, new_goals)
        if sub_plan is None:
            return None

        parallel_actions = {a for a in selected if a is not None}
        if parallel_actions:
            sub_plan.append(parallel_actions)

        return sub_plan

    def _select_achievers(
        self,
        goal_achievers: Dict[str, List[Optional[Action]]],
        layer: Layer
    ) -> Optional[List[Optional[Action]]]:
        goals = list(goal_achievers.keys())
        selected = []

        for goal in goals:
            achievers = goal_achievers[goal]
            found = False

            for achiever in achievers:
                conflict = False
                for prev_achiever in selected:
                    if achiever is None or prev_achiever is None:
                        continue
                    if frozenset({achiever, prev_achiever}) in layer.action_mutex:
                        conflict = True
                        break

                if not conflict:
                    selected.append(achiever)
                    found = True
                    break

            if not found:
                return None

        return selected

# ============================================================================
# Failure Information
# ============================================================================

@dataclass
class FailureInfo:
    """Information about a detected failure"""
    
    is_failure: bool = False
    failure_type: Optional[FailureType] = None
    agent_name: Optional[str] = None
    node_id: Optional[str] = None
    
    # Details
    error_message: Optional[str] = None
    confidence: float = 0.0
    evidence: Optional[str] = None
    
    # Context
    input_state: Optional[Dict[str, Any]] = None
    output: Optional[Any] = None
    
    # Timing
    detected_at: datetime = field(default_factory=datetime.now)
    
    def __bool__(self):
        return self.is_failure
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_failure": self.is_failure,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "agent_name": self.agent_name,
            "error_message": self.error_message,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "detected_at": self.detected_at.isoformat()
        }


# ============================================================================
# Repair Results
# ============================================================================

@dataclass
class RepairResult:
    """Result of a repair attempt"""
    
    success: bool = False
    strategy_used: Optional[str] = None
    
    # Output
    new_output: Optional[Any] = None
    
    # Diagnostics
    attempts: int = 0
    reason: Optional[str] = None
    modifications_made: List[str] = field(default_factory=list)
    
    # Timing
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "strategy_used": self.strategy_used,
            "attempts": self.attempts,
            "reason": self.reason,
            "modifications_made": self.modifications_made,
            "latency_ms": self.latency_ms
        }


# ============================================================================
# Recomposition Results
# ============================================================================

@dataclass
class RecomposeResult:
    """Result of a workflow recomposition"""
    
    success: bool = False
    
    # New workflow
    new_workflow: Optional[Any] = None  # The recomposed StateGraph
    new_workflow_spec: Optional[Dict[str, Any]] = None  # JSON specification
    
    # Changes made
    nodes_added: List[str] = field(default_factory=list)
    nodes_removed: List[str] = field(default_factory=list)
    edges_changed: List[Dict[str, str]] = field(default_factory=list)
    
    # Reasoning
    analysis: Optional[str] = None
    rationale: Optional[str] = None
    
    # Timing
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "nodes_added": self.nodes_added,
            "nodes_removed": self.nodes_removed,
            "edges_changed": self.edges_changed,
            "analysis": self.analysis,
            "rationale": self.rationale,
            "latency_ms": self.latency_ms
        }


# ============================================================================
# Healing Log
# ============================================================================

@dataclass
class HealingEvent:
    """A single healing event in the log"""
    
    timestamp: datetime
    event_type: str  # "detection", "repair_attempt", "repair_success", "recompose", etc.
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "details": self.details
        }


@dataclass
class HealingLog:
    """Complete log of healing activities for a workflow run"""
    
    run_id: str
    started_at: datetime = field(default_factory=datetime.now)
    
    # Failures detected
    failures_detected: List[FailureInfo] = field(default_factory=list)
    
    # Recovery attempts
    repair_attempts: int = 0
    repair_successes: int = 0
    recompose_attempts: int = 0
    recompose_successes: int = 0
    
    # Events
    events: List[HealingEvent] = field(default_factory=list)
    
    # Final status
    final_status: str = "pending"  # "success", "failed", "escalated"
    
    def add_event(self, event_type: str, details: Dict[str, Any]):
        """Add an event to the log"""
        self.events.append(HealingEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            details=details
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "failures_detected": [f.to_dict() for f in self.failures_detected],
            "repair_attempts": self.repair_attempts,
            "repair_successes": self.repair_successes,
            "recompose_attempts": self.recompose_attempts,
            "recompose_successes": self.recompose_successes,
            "events": [e.to_dict() for e in self.events],
            "final_status": self.final_status
        }


# ============================================================================
# Workflow State (for LangGraph)
# ============================================================================

class AEGISState(TypedDict, total=False):
    """
    Extended state for AEGIS-wrapped workflows.
    Includes original workflow state plus AEGIS metadata.
    """
    
    # Original workflow data (will be extended by specific workflows)
    # ... user's workflow state goes here ...
    
    # AEGIS metadata
    aegis_enabled: bool
    aegis_healing_log: Dict[str, Any]
    aegis_current_failures: List[Dict[str, Any]]
    aegis_recovery_mode: bool
    aegis_original_input: Dict[str, Any]


def create_aegis_state(base_state_class: type) -> type:
    """
    Factory function to create an AEGIS-enhanced state class
    that extends a user's workflow state.
    
    Usage:
        class MyWorkflowState(TypedDict):
            topic: str
            result: str
        
        AEGISMyWorkflowState = create_aegis_state(MyWorkflowState)
    """
    
    class CombinedState(base_state_class, total=False):
        aegis_enabled: bool
        aegis_healing_log: Dict[str, Any]
        aegis_current_failures: List[Dict[str, Any]]
        aegis_recovery_mode: bool
        aegis_original_input: Dict[str, Any]
    
    return CombinedState


# ============================================================================
# Agent Information
# ============================================================================

@dataclass
class AgentInfo:
    """Information about a registered agent"""
    
    name: str
    description: str
    
    # Capabilities
    role: str  # "researcher", "analyzer", "writer", "validator", etc.
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    
    # The actual agent function
    agent_func: Optional[callable] = None
    
    # Configuration
    default_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    
    # Performance metadata
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "tools": self.tools,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate
        }
