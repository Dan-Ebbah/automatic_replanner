"""
AEGIS Detector Module
=====================
Detects failures in agent outputs including crashes, hallucinations,
semantic drift, and format errors.
"""

import json
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import asyncio
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .config import AEGISConfig, FailureType, LLMProvider, DetectorConfig
from .state import FailureInfo


class AEGISDetector:
    """
    Detects various types of failures in agent outputs.

    Failure Types Detected:
    - CRASH: Exceptions and runtime errors
    - TIMEOUT: Agent took too long to respond
    - HALLUCINATION: Output contains made-up information
    - SEMANTIC_DRIFT: Output doesn't align with the task
    - FORMAT_ERROR: Output doesn't match expected schema
    - QUALITY: Output is too short, empty, or low quality
    """

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from text that may be wrapped in markdown code blocks."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code blocks
        if "```json" in text:
            try:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass

        if "```" in text:
            try:
                json_str = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass

        # Try to find JSON object in text
        import re
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("Could not extract JSON from response", text, 0)

    def __init__(self, config: AEGISConfig):
        self.config = config
        self.detector_config = config.detector
        self.llm = self._create_llm()
        
        # Validators to run (in order)
        self.validators: List[Callable] = [
            self._check_crash,
            self._check_empty_output,
            self._check_format,
            self._check_hallucination,
            self._check_semantic_alignment,
            self._check_quality
        ]
    
    def _create_llm(self):
        """Create the LLM instance for semantic checks"""
        if self.config.llm_provider == LLMProvider.OPENAI:
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=0.1,  # Low temp for consistent detection
                api_key=self.config.openai_api_key
            )
        elif self.config.llm_provider == LLMProvider.ANTHROPIC:
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=0.1,
                api_key=self.config.anthropic_api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
    def detect(
        self,
        agent_name: str,
        agent_output: Any,
        original_task: str,
        input_state: Dict[str, Any],
        expected_schema: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> FailureInfo:
        """
        Run all validators on agent output and return failure info if any.
        
        Args:
            agent_name: Name of the agent that produced the output
            agent_output: The output from the agent
            original_task: The task the agent was supposed to perform
            input_state: The input state passed to the agent
            expected_schema: Optional JSON schema the output should match
            context: Additional context for semantic validation
        
        Returns:
            FailureInfo object (check .is_failure to see if failure detected)
        """
        
        detection_context = {
            "agent_name": agent_name,
            "agent_output": agent_output,
            "original_task": original_task,
            "input_state": input_state,
            "expected_schema": expected_schema,
            "context": context
        }
        
        # Run each validator
        for validator in self.validators:
            try:
                result = validator(detection_context)
                if result.is_failure:
                    return result
            except Exception as e:
                # Validator itself failed - log but continue
                print(f"Warning: Validator {validator.__name__} failed: {e}")
                continue
        
        # No failures detected
        return FailureInfo(is_failure=False)
    
    def _check_crash(self, ctx: Dict[str, Any]) -> FailureInfo:
        """Check if the output indicates a crash/exception"""
        
        output = ctx["agent_output"]
        
        # Check if output is an exception
        if isinstance(output, Exception):
            return FailureInfo(
                is_failure=True,
                failure_type=FailureType.CRASH,
                agent_name=ctx["agent_name"],
                error_message=str(output),
                confidence=1.0,
                evidence=f"Exception raised: {type(output).__name__}",
                input_state=ctx["input_state"],
                output=output
            )
        
        # Check if output is a dict with error field
        if isinstance(output, dict):
            if "error" in output or "exception" in output:
                return FailureInfo(
                    is_failure=True,
                    failure_type=FailureType.CRASH,
                    agent_name=ctx["agent_name"],
                    error_message=output.get("error") or output.get("exception"),
                    confidence=0.9,
                    evidence="Error field present in output",
                    input_state=ctx["input_state"],
                    output=output
                )
        
        return FailureInfo(is_failure=False)
    
    def _check_empty_output(self, ctx: Dict[str, Any]) -> FailureInfo:
        """Check if output is empty or None"""
        
        output = ctx["agent_output"]
        
        if output is None:
            return FailureInfo(
                is_failure=True,
                failure_type=FailureType.QUALITY,
                agent_name=ctx["agent_name"],
                error_message="Agent returned None",
                confidence=1.0,
                evidence="Output is None",
                input_state=ctx["input_state"],
                output=output
            )
        
        # Check for empty strings/dicts/lists
        if isinstance(output, (str, dict, list)) and len(output) == 0:
            return FailureInfo(
                is_failure=True,
                failure_type=FailureType.QUALITY,
                agent_name=ctx["agent_name"],
                error_message="Agent returned empty output",
                confidence=1.0,
                evidence=f"Output is empty {type(output).__name__}",
                input_state=ctx["input_state"],
                output=output
            )
        
        return FailureInfo(is_failure=False)
    
    def _check_format(self, ctx: Dict[str, Any]) -> FailureInfo:
        """Check if output matches expected schema"""
        
        if not self.detector_config.schema_validation_enabled:
            return FailureInfo(is_failure=False)
        
        expected_schema = ctx.get("expected_schema")
        if not expected_schema:
            return FailureInfo(is_failure=False)
        
        output = ctx["agent_output"]
        
        # Basic type checking
        expected_type = expected_schema.get("type")
        if expected_type:
            type_mapping = {
                "string": str,
                "object": dict,
                "array": list,
                "number": (int, float),
                "boolean": bool
            }
            expected_python_type = type_mapping.get(expected_type)
            if expected_python_type and not isinstance(output, expected_python_type):
                return FailureInfo(
                    is_failure=True,
                    failure_type=FailureType.FORMAT_ERROR,
                    agent_name=ctx["agent_name"],
                    error_message=f"Expected {expected_type}, got {type(output).__name__}",
                    confidence=1.0,
                    evidence=f"Type mismatch",
                    input_state=ctx["input_state"],
                    output=output
                )
        
        # Check required fields for objects
        if isinstance(output, dict) and "required" in expected_schema:
            missing = [f for f in expected_schema["required"] if f not in output]
            if missing:
                return FailureInfo(
                    is_failure=True,
                    failure_type=FailureType.FORMAT_ERROR,
                    agent_name=ctx["agent_name"],
                    error_message=f"Missing required fields: {missing}",
                    confidence=1.0,
                    evidence=f"Required fields not present",
                    input_state=ctx["input_state"],
                    output=output
                )
        
        return FailureInfo(is_failure=False)
    
    def _check_hallucination(self, ctx: Dict[str, Any]) -> FailureInfo:
        """Use LLM to detect hallucinations in the output"""
        
        if not self.detector_config.hallucination_check_enabled:
            return FailureInfo(is_failure=False)
        
        output = ctx["agent_output"]
        if not isinstance(output, str):
            output = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
        
        # Skip if output is too short
        if len(output) < 50:
            return FailureInfo(is_failure=False)
        
        prompt = f"""Analyze the following output for hallucinations (made-up information).

TASK: {ctx["original_task"]}

INPUT CONTEXT: {json.dumps(ctx["input_state"], indent=2)[:1000]}

OUTPUT TO CHECK:
{output[:2000]}

Analyze if this output contains:
1. Made-up facts not grounded in the input
2. Invented names, dates, numbers, or statistics
3. Claims that contradict the provided context
4. References to non-existent sources or quotes

Respond with JSON only:
{{
    "has_hallucination": true/false,
    "confidence": 0.0-1.0,
    "evidence": "specific examples of hallucinated content or 'none found'",
    "reasoning": "brief explanation"
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = self._extract_json(response.content)

            if result.get("has_hallucination", False):
                confidence = result.get("confidence", 0.5)
                if confidence >= self.detector_config.hallucination_confidence_threshold:
                    return FailureInfo(
                        is_failure=True,
                        failure_type=FailureType.HALLUCINATION,
                        agent_name=ctx["agent_name"],
                        error_message="Hallucination detected in output",
                        confidence=confidence,
                        evidence=result.get("evidence", ""),
                        input_state=ctx["input_state"],
                        output=ctx["agent_output"]
                    )
        except Exception as e:
            print(f"Warning: Hallucination check failed: {e}")

        return FailureInfo(is_failure=False)

    def _check_semantic_alignment(self, ctx: Dict[str, Any]) -> FailureInfo:
        """Check if output semantically aligns with the task"""
        
        if not self.detector_config.semantic_check_enabled:
            return FailureInfo(is_failure=False)
        
        output = ctx["agent_output"]
        if not isinstance(output, str):
            output = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
        
        prompt = f"""Rate how well this output addresses the given task.

TASK: {ctx["original_task"]}

OUTPUT:
{output[:2000]}

Rate the semantic alignment:
1 = Completely off-topic, doesn't address the task at all
2 = Partially relevant but misses the main point
3 = Addresses the task but with significant issues
4 = Good alignment with minor issues
5 = Perfect alignment, fully addresses the task

Respond with JSON only:
{{
    "alignment_score": 1-5,
    "issues": ["list of specific issues if any"],
    "reasoning": "brief explanation"
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = self._extract_json(response.content)

            score = result.get("alignment_score", 5)
            # Convert 1-5 score to 0-1 scale
            normalized_score = (score - 1) / 4.0

            if normalized_score < self.detector_config.semantic_alignment_threshold:
                return FailureInfo(
                    is_failure=True,
                    failure_type=FailureType.SEMANTIC_DRIFT,
                    agent_name=ctx["agent_name"],
                    error_message=f"Output doesn't align with task (score: {score}/5)",
                    confidence=1.0 - normalized_score,
                    evidence="; ".join(result.get("issues", [])),
                    input_state=ctx["input_state"],
                    output=ctx["agent_output"]
                )
        except Exception as e:
            print(f"Warning: Semantic alignment check failed: {e}")
        
        return FailureInfo(is_failure=False)
    
    def _check_quality(self, ctx: Dict[str, Any]) -> FailureInfo:
        """Check basic quality metrics of the output"""
        
        if not self.detector_config.quality_check_enabled:
            return FailureInfo(is_failure=False)
        
        output = ctx["agent_output"]
        
        # Convert to string for length check
        if isinstance(output, str):
            output_str = output
        elif isinstance(output, (dict, list)):
            output_str = json.dumps(output)
        else:
            output_str = str(output)
        
        # Check minimum length
        if len(output_str) < self.detector_config.min_output_length:
            return FailureInfo(
                is_failure=True,
                failure_type=FailureType.QUALITY,
                agent_name=ctx["agent_name"],
                error_message=f"Output too short ({len(output_str)} chars)",
                confidence=0.8,
                evidence=f"Minimum required: {self.detector_config.min_output_length}",
                input_state=ctx["input_state"],
                output=ctx["agent_output"]
            )
        
        # Check maximum length (might indicate runaway generation)
        if len(output_str) > self.detector_config.max_output_length:
            return FailureInfo(
                is_failure=True,
                failure_type=FailureType.QUALITY,
                agent_name=ctx["agent_name"],
                error_message=f"Output too long ({len(output_str)} chars)",
                confidence=0.7,
                evidence=f"Maximum allowed: {self.detector_config.max_output_length}",
                input_state=ctx["input_state"],
                output=ctx["agent_output"]
            )
        
        return FailureInfo(is_failure=False)
    
    async def detect_async(
        self,
        agent_name: str,
        agent_output: Any,
        original_task: str,
        input_state: Dict[str, Any],
        expected_schema: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> FailureInfo:
        """Async version of detect()"""
        # For now, wrap sync in async - can be optimized later
        return await asyncio.to_thread(
            self.detect,
            agent_name,
            agent_output,
            original_task,
            input_state,
            expected_schema,
            context
        )
