"""
AEGIS Wrapper
=============
Main wrapper that integrates detection, repair, and recomposition
to create self-healing LangGraph workflows.
"""

import uuid
import time
import json
import functools
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union
from datetime import datetime
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

from .config import AEGISConfig, FailureType, RecoveryStrategy, default_config
from .state import (
    FailureInfo, RepairResult, RecomposeResult, HealingLog, HealingEvent,
    AEGISState, create_aegis_state
)
from .detector import AEGISDetector
from .repair import AEGISRepair
from .recompose import AEGISRecompose
from .registry import AgentRegistry, default_registry


class AEGIS:
    """
    AEGIS: Autonomous Error-handling and Graph-recomposition 
    for Intelligent agent Systems
    
    A drop-in wrapper that adds self-healing capabilities to any 
    LangGraph workflow.
    
    Usage:
        # Create your workflow
        workflow = StateGraph(MyState)
        workflow.add_node("agent1", agent1_func)
        ...
        compiled = workflow.compile()
        
        # Wrap with AEGIS
        aegis_workflow = AEGIS.wrap(compiled)
        
        # Run with self-healing
        result = aegis_workflow.invoke({"input": "..."})
    """
    
    def __init__(
        self,
        config: Optional[AEGISConfig] = None,
        agent_registry: Optional[AgentRegistry] = None
    ):
        self.config = config or default_config
        self.registry = agent_registry or default_registry
        
        # Initialize components
        self.detector = AEGISDetector(self.config)
        self.repair = AEGISRepair(self.config, self.detector)
        self.recompose = AEGISRecompose(self.config, self.registry)
        
        # Tracking
        self.healing_logs: Dict[str, HealingLog] = {}
    
    @classmethod
    def wrap(
        cls,
        workflow: Union[StateGraph, CompiledGraph],
        config: Optional[AEGISConfig] = None,
        agent_registry: Optional[AgentRegistry] = None
    ) -> "AEGISWorkflow":
        """
        Wrap a LangGraph workflow with AEGIS self-healing capabilities.
        
        Args:
            workflow: A LangGraph StateGraph or CompiledGraph
            config: Optional AEGIS configuration
            agent_registry: Optional agent registry for recomposition
        
        Returns:
            AEGISWorkflow with self-healing capabilities
        """
        aegis = cls(config, agent_registry)
        return AEGISWorkflow(workflow, aegis)
    
    def create_healing_log(self) -> HealingLog:
        """Create a new healing log for a workflow run"""
        run_id = str(uuid.uuid4())
        log = HealingLog(run_id=run_id)
        self.healing_logs[run_id] = log
        return log
    
    def detect_failure(
        self,
        agent_name: str,
        agent_output: Any,
        task: str,
        input_state: Dict[str, Any],
        expected_schema: Optional[Dict[str, Any]] = None
    ) -> FailureInfo:
        """Detect failures in agent output"""
        return self.detector.detect(
            agent_name=agent_name,
            agent_output=agent_output,
            original_task=task,
            input_state=input_state,
            expected_schema=expected_schema
        )
    
    def attempt_repair(
        self,
        failure_info: FailureInfo,
        agent_func: Callable,
        prompt: str,
        input_state: Dict[str, Any],
        task: str
    ) -> RepairResult:
        """Attempt to repair a failed agent output"""
        return self.repair.repair(
            failure_info=failure_info,
            agent_func=agent_func,
            original_prompt=prompt,
            input_state=input_state,
            original_task=task
        )
    
    def attempt_recompose(
        self,
        workflow_spec: Dict[str, Any],
        failure_info: FailureInfo,
        task: str,
        history: List[Dict[str, Any]],
        repair_attempts: int
    ) -> RecomposeResult:
        """Attempt to recompose the workflow"""
        return self.recompose.recompose(
            workflow_spec=workflow_spec,
            failure_info=failure_info,
            original_task=task,
            execution_history=history,
            repair_attempts=repair_attempts
        )


class AEGISWorkflow:
    """
    A wrapped LangGraph workflow with self-healing capabilities.
    """
    
    def __init__(
        self,
        workflow: Union[StateGraph, CompiledGraph],
        aegis: AEGIS
    ):
        self.aegis = aegis
        self.config = aegis.config
        
        # Handle both compiled and uncompiled workflows
        if isinstance(workflow, StateGraph):
            self.original_graph = workflow
            self.compiled_workflow = workflow.compile()
        else:
            self.original_graph = None
            self.compiled_workflow = workflow
        
        # Extract workflow structure
        self.workflow_spec = self._extract_workflow_spec()
        
        # Runtime state
        self.current_log: Optional[HealingLog] = None
        self.execution_history: List[Dict[str, Any]] = []
    
    def _extract_workflow_spec(self) -> Dict[str, Any]:
        """Extract workflow specification from compiled graph"""
        
        # This is a simplified extraction - in practice, you'd want
        # to properly introspect the LangGraph structure
        try:
            nodes = []
            edges = []
            
            # Try to extract from graph structure
            if hasattr(self.compiled_workflow, 'nodes'):
                for node_name in self.compiled_workflow.nodes:
                    if node_name not in ('__start__', '__end__'):
                        nodes.append({
                            "id": node_name,
                            "agent": node_name,
                            "config": {}
                        })
            
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            return {"nodes": [], "edges": []}
    
    def invoke(
        self,
        input_state: Dict[str, Any],
        task_description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke the workflow with self-healing enabled.
        
        Args:
            input_state: Input state for the workflow
            task_description: Optional description of the task (for semantic checking)
            config: Optional LangGraph config
        
        Returns:
            Final output state (with healing applied if needed)
        """
        
        # Initialize healing log
        self.current_log = self.aegis.create_healing_log()
        self.execution_history = []
        
        # Extract task description if not provided
        if not task_description:
            task_description = self._infer_task(input_state)
        
        self.current_log.add_event("workflow_started", {
            "input": self._safe_serialize(input_state),
            "task": task_description
        })
        
        try:
            # Run the workflow with monitoring
            result = self._execute_with_healing(
                input_state=input_state,
                task=task_description,
                config=config
            )
            
            self.current_log.final_status = "success"
            self.current_log.add_event("workflow_completed", {
                "output": self._safe_serialize(result)
            })
            
            return result
            
        except Exception as e:
            self.current_log.final_status = "failed"
            self.current_log.add_event("workflow_failed", {
                "error": str(e)
            })
            raise
    
    def _execute_with_healing(
        self,
        input_state: Dict[str, Any],
        task: str,
        config: Optional[Dict[str, Any]] = None,
        current_workflow: Optional[CompiledGraph] = None
    ) -> Dict[str, Any]:
        """Execute workflow with healing loop"""
        
        workflow = current_workflow or self.compiled_workflow
        repair_attempts = 0
        recompose_attempts = 0
        
        while True:
            try:
                # Attempt to run the workflow
                result = workflow.invoke(input_state, config=config)
                
                # Check the final output for semantic issues
                failure = self._check_final_output(result, task, input_state)
                
                if not failure.is_failure:
                    return result
                
                # Failure detected - attempt recovery
                self.current_log.failures_detected.append(failure)
                self.current_log.add_event("failure_detected", failure.to_dict())
                
                # Try repair first
                if repair_attempts < self.config.repair.max_repair_attempts:
                    repair_result = self._attempt_repair(failure, input_state, task)
                    repair_attempts += 1
                    self.current_log.repair_attempts += 1
                    
                    if repair_result.success:
                        self.current_log.repair_successes += 1
                        # Update the result with repaired output
                        if isinstance(result, dict) and repair_result.new_output:
                            result.update({"_repaired_output": repair_result.new_output})
                        return result
                
                # Try recomposition
                if (self.config.recompose.enable_recomposition and 
                    recompose_attempts < self.config.recompose.max_recomposition_attempts):
                    
                    recompose_result = self.aegis.attempt_recompose(
                        workflow_spec=self.workflow_spec,
                        failure_info=failure,
                        task=task,
                        history=self.execution_history,
                        repair_attempts=repair_attempts
                    )
                    recompose_attempts += 1
                    self.current_log.recompose_attempts += 1
                    
                    if recompose_result.success:
                        self.current_log.recompose_successes += 1
                        self.current_log.add_event("workflow_recomposed", 
                            recompose_result.to_dict())
                        
                        # Update workflow spec
                        self.workflow_spec = recompose_result.new_workflow_spec
                        workflow = recompose_result.new_workflow
                        
                        # Retry with new workflow
                        repair_attempts = 0  # Reset repair counter
                        continue
                
                # All recovery attempts exhausted
                self.current_log.add_event("recovery_exhausted", {
                    "repair_attempts": repair_attempts,
                    "recompose_attempts": recompose_attempts
                })
                
                # Return best effort result
                return result
                
            except Exception as e:
                # Handle crashes
                failure = FailureInfo(
                    is_failure=True,
                    failure_type=FailureType.CRASH,
                    error_message=str(e),
                    input_state=input_state
                )
                
                self.current_log.failures_detected.append(failure)
                
                # Try to recover from crash
                if repair_attempts < self.config.repair.max_repair_attempts:
                    repair_attempts += 1
                    self.current_log.repair_attempts += 1
                    # For crashes, we might retry with simplified input
                    continue
                
                raise
    
    def _check_final_output(
        self,
        output: Any,
        task: str,
        input_state: Dict[str, Any]
    ) -> FailureInfo:
        """Check the final workflow output for issues"""
        
        return self.aegis.detect_failure(
            agent_name="workflow_output",
            agent_output=output,
            task=task,
            input_state=input_state
        )
    
    def _attempt_repair(
        self,
        failure: FailureInfo,
        input_state: Dict[str, Any],
        task: str
    ) -> RepairResult:
        """Attempt to repair a failure"""
        
        # For now, we use a generic repair approach
        # In a full implementation, we'd identify the specific agent that failed
        
        return self.aegis.attempt_repair(
            failure_info=failure,
            agent_func=lambda x: x,  # Placeholder
            prompt=task,
            input_state=input_state,
            task=task
        )
    
    def _infer_task(self, input_state: Dict[str, Any]) -> str:
        """Infer task description from input state"""
        
        # Look for common task-describing fields
        for field in ["task", "query", "question", "prompt", "input", "topic"]:
            if field in input_state:
                return str(input_state[field])
        
        # Fall back to serializing input
        return f"Process input: {self._safe_serialize(input_state)[:200]}"
    
    def _safe_serialize(self, obj: Any) -> str:
        """Safely serialize an object to string"""
        try:
            return json.dumps(obj, default=str)
        except:
            return str(obj)
    
    def get_healing_log(self) -> Optional[HealingLog]:
        """Get the healing log from the last run"""
        return self.current_log
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from the last run"""
        if not self.current_log:
            return {}
        
        return {
            "failures_detected": len(self.current_log.failures_detected),
            "repair_attempts": self.current_log.repair_attempts,
            "repair_successes": self.current_log.repair_successes,
            "repair_success_rate": (
                self.current_log.repair_successes / self.current_log.repair_attempts
                if self.current_log.repair_attempts > 0 else 0
            ),
            "recompose_attempts": self.current_log.recompose_attempts,
            "recompose_successes": self.current_log.recompose_successes,
            "final_status": self.current_log.final_status
        }


# Convenience function for wrapping agents
def with_aegis_monitoring(
    agent_name: str,
    task: str,
    aegis: Optional[AEGIS] = None
):
    """
    Decorator to add AEGIS monitoring to individual agent functions.
    
    Usage:
        @with_aegis_monitoring("research_agent", "Research the topic")
        def research_agent(state):
            ...
    """
    
    _aegis = aegis or AEGIS()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            
            try:
                result = func(state)
                
                # Check for failures
                failure = _aegis.detect_failure(
                    agent_name=agent_name,
                    agent_output=result,
                    task=task,
                    input_state=state
                )
                
                if failure.is_failure:
                    # Attempt repair
                    repair_result = _aegis.attempt_repair(
                        failure_info=failure,
                        agent_func=func,
                        prompt=task,
                        input_state=state,
                        task=task
                    )
                    
                    if repair_result.success:
                        return repair_result.new_output
                
                return result
                
            except Exception as e:
                # Log and re-raise
                raise
        
        return wrapper
    
    return decorator
