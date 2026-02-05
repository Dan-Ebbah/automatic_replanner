"""
AEGIS Repair Module
===================
Repairs failed agent outputs using various strategies including
prompt enhancement, grounding, and regeneration.
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

from .config import AEGISConfig, FailureType, LLMProvider, RepairConfig
from .state import FailureInfo, RepairResult


class AEGISRepair:
    """
    Repairs failed agent outputs using multiple strategies.
    
    Repair Strategies:
    - Prompt Enhancement: Add clarifications to prevent the error
    - Output Regeneration: Re-run with different parameters
    - Grounding Injection: Add facts to prevent hallucination
    - Schema Enforcement: Force output to match expected format
    - Temperature Adjustment: Reduce randomness for more consistent output
    """
    
    def __init__(self, config: AEGISConfig, detector=None):
        self.config = config
        self.repair_config = config.repair
        self.llm = self._create_llm()
        self.detector = detector  # Optional reference to detector for verification
        
        # Strategy mapping
        self.strategies: Dict[FailureType, Callable] = {
            FailureType.HALLUCINATION: self._repair_hallucination,
            FailureType.SEMANTIC_DRIFT: self._repair_semantic_drift,
            FailureType.FORMAT_ERROR: self._repair_format_error,
            FailureType.QUALITY: self._repair_quality,
            FailureType.CRASH: self._repair_crash,
            FailureType.TIMEOUT: self._repair_timeout,
        }
    
    def _create_llm(self):
        """Create the LLM instance for repair operations"""
        if self.config.llm_provider == LLMProvider.OPENAI:
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=0.2,
                api_key=self.config.openai_api_key
            )
        elif self.config.llm_provider == LLMProvider.ANTHROPIC:
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=0.2,
                api_key=self.config.anthropic_api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
    def repair(
        self,
        failure_info: FailureInfo,
        agent_func: Callable,
        original_prompt: str,
        input_state: Dict[str, Any],
        original_task: str,
        context: Optional[str] = None
    ) -> RepairResult:
        """
        Attempt to repair a failed agent output.
        
        Args:
            failure_info: Information about the detected failure
            agent_func: The original agent function to re-invoke
            original_prompt: The original prompt used
            input_state: The input state for the agent
            original_task: The task description
            context: Additional context for repair
        
        Returns:
            RepairResult with success status and new output if successful
        """
        
        start_time = time.time()
        
        # Get appropriate repair strategy
        strategy = self.strategies.get(failure_info.failure_type)
        if not strategy:
            return RepairResult(
                success=False,
                reason=f"No repair strategy for failure type: {failure_info.failure_type}",
                latency_ms=(time.time() - start_time) * 1000
            )
        
        # Try repair with multiple attempts
        for attempt in range(self.repair_config.max_repair_attempts):
            try:
                result = strategy(
                    failure_info=failure_info,
                    agent_func=agent_func,
                    original_prompt=original_prompt,
                    input_state=input_state,
                    original_task=original_task,
                    context=context,
                    attempt=attempt
                )
                
                result.attempts = attempt + 1
                result.latency_ms = (time.time() - start_time) * 1000
                
                if result.success:
                    return result
                
                # Brief backoff before next attempt
                if attempt < self.repair_config.max_repair_attempts - 1:
                    time.sleep(self.repair_config.repair_backoff_seconds)
                    
            except Exception as e:
                print(f"Repair attempt {attempt + 1} failed with error: {e}")
                continue
        
        # All attempts failed
        return RepairResult(
            success=False,
            attempts=self.repair_config.max_repair_attempts,
            reason="All repair attempts exhausted",
            latency_ms=(time.time() - start_time) * 1000
        )
    
    def _repair_hallucination(
        self,
        failure_info: FailureInfo,
        agent_func: Callable,
        original_prompt: str,
        input_state: Dict[str, Any],
        original_task: str,
        context: Optional[str],
        attempt: int
    ) -> RepairResult:
        """Repair hallucination by grounding in facts"""
        
        modifications = []
        
        # Step 1: Extract/retrieve relevant facts
        facts = self._retrieve_grounding_facts(input_state, original_task, context)
        modifications.append("Retrieved grounding facts")
        
        # Step 2: Create enhanced prompt with grounding
        enhanced_prompt = f"""{original_prompt}

IMPORTANT GROUNDING INSTRUCTIONS:
- Base your response ONLY on the information provided below
- Do NOT invent, assume, or fabricate any facts
- If information is not available, explicitly state "Information not provided"
- Do NOT make up names, dates, statistics, or quotes

VERIFIED FACTS TO USE:
{facts}

PREVIOUS ATTEMPT FAILED DUE TO: {failure_info.evidence}
Please correct this in your new response."""
        
        modifications.append("Added grounding instructions")
        
        # Step 3: Reduce temperature for more consistent output
        if self.repair_config.enable_temperature_adjustment:
            temp = self.repair_config.retry_temperatures[min(attempt, len(self.repair_config.retry_temperatures)-1)]
            modifications.append(f"Reduced temperature to {temp}")
        
        # Step 4: Regenerate with enhanced prompt
        try:
            new_output = self._regenerate_with_prompt(enhanced_prompt, input_state, attempt)
            
            # Step 5: Verify the fix (if detector available)
            if self.detector:
                recheck = self.detector.detect(
                    agent_name=failure_info.agent_name,
                    agent_output=new_output,
                    original_task=original_task,
                    input_state=input_state
                )
                if recheck.is_failure:
                    return RepairResult(
                        success=False,
                        reason=f"Repair verification failed: {recheck.error_message}",
                        new_output=new_output,
                        modifications_made=modifications
                    )
            
            return RepairResult(
                success=True,
                strategy_used="hallucination_grounding",
                new_output=new_output,
                modifications_made=modifications
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                reason=f"Regeneration failed: {str(e)}",
                modifications_made=modifications
            )
    
    def _repair_semantic_drift(
        self,
        failure_info: FailureInfo,
        agent_func: Callable,
        original_prompt: str,
        input_state: Dict[str, Any],
        original_task: str,
        context: Optional[str],
        attempt: int
    ) -> RepairResult:
        """Repair semantic drift by clarifying the task"""
        
        modifications = []
        
        # Step 1: Analyze what went wrong
        analysis_prompt = f"""The following output did not properly address the task.

TASK: {original_task}
OUTPUT: {failure_info.output}
ISSUES: {failure_info.evidence}

Explain specifically:
1. What the task required
2. What the output actually did
3. What corrections are needed

Respond in JSON:
{{
    "task_requirements": ["list of key requirements"],
    "output_problems": ["what was wrong"],
    "corrections_needed": ["specific fixes"]
}}"""

        try:
            analysis_response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            analysis = json.loads(analysis_response.content)
            modifications.append("Analyzed semantic drift cause")
        except:
            analysis = {"corrections_needed": ["Re-read the task carefully and ensure output directly addresses it"]}
        
        # Step 2: Create clarified prompt
        enhanced_prompt = f"""{original_prompt}

CRITICAL TASK CLARIFICATION:
The task requires: {original_task}

Your response MUST:
{chr(10).join(f"- {req}" for req in analysis.get("task_requirements", ["Address the task directly"]))}

AVOID these issues from previous attempt:
{chr(10).join(f"- {prob}" for prob in analysis.get("output_problems", []))}

Corrections needed:
{chr(10).join(f"- {corr}" for corr in analysis.get("corrections_needed", []))}"""

        modifications.append("Added task clarifications")
        
        # Step 3: Regenerate
        try:
            new_output = self._regenerate_with_prompt(enhanced_prompt, input_state, attempt)
            
            return RepairResult(
                success=True,
                strategy_used="semantic_clarification",
                new_output=new_output,
                modifications_made=modifications
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                reason=f"Regeneration failed: {str(e)}",
                modifications_made=modifications
            )
    
    def _repair_format_error(
        self,
        failure_info: FailureInfo,
        agent_func: Callable,
        original_prompt: str,
        input_state: Dict[str, Any],
        original_task: str,
        context: Optional[str],
        attempt: int
    ) -> RepairResult:
        """Repair format errors by enforcing schema"""
        
        modifications = []
        
        # Try to parse and fix the output directly first
        if attempt == 0:
            try:
                fixed = self._attempt_format_fix(failure_info.output, failure_info.error_message)
                if fixed:
                    modifications.append("Direct format fix applied")
                    return RepairResult(
                        success=True,
                        strategy_used="direct_format_fix",
                        new_output=fixed,
                        modifications_made=modifications
                    )
            except:
                pass
        
        # Create schema-enforcing prompt
        enhanced_prompt = f"""{original_prompt}

OUTPUT FORMAT REQUIREMENTS:
Your response MUST be valid JSON matching this structure.
{failure_info.evidence}

Previous attempt failed because: {failure_info.error_message}

Respond with ONLY the JSON, no markdown code blocks or explanations."""

        modifications.append("Added schema enforcement")
        
        try:
            new_output = self._regenerate_with_prompt(enhanced_prompt, input_state, attempt)
            
            # Try to parse as JSON if expected
            if isinstance(new_output, str):
                try:
                    new_output = json.loads(new_output.strip().strip('```json').strip('```'))
                    modifications.append("Parsed JSON output")
                except:
                    pass
            
            return RepairResult(
                success=True,
                strategy_used="schema_enforcement",
                new_output=new_output,
                modifications_made=modifications
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                reason=f"Format repair failed: {str(e)}",
                modifications_made=modifications
            )
    
    def _repair_quality(
        self,
        failure_info: FailureInfo,
        agent_func: Callable,
        original_prompt: str,
        input_state: Dict[str, Any],
        original_task: str,
        context: Optional[str],
        attempt: int
    ) -> RepairResult:
        """Repair quality issues (too short, too long, empty)"""
        
        modifications = []
        
        enhanced_prompt = f"""{original_prompt}

QUALITY REQUIREMENTS:
- Provide a complete, thorough response
- Minimum several sentences of substantive content
- Maximum reasonable length (don't be overly verbose)
- Ensure all key points are addressed

Previous attempt issue: {failure_info.error_message}"""

        modifications.append("Added quality requirements")
        
        try:
            new_output = self._regenerate_with_prompt(enhanced_prompt, input_state, attempt)
            
            return RepairResult(
                success=True,
                strategy_used="quality_enhancement",
                new_output=new_output,
                modifications_made=modifications
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                reason=f"Quality repair failed: {str(e)}",
                modifications_made=modifications
            )
    
    def _repair_crash(
        self,
        failure_info: FailureInfo,
        agent_func: Callable,
        original_prompt: str,
        input_state: Dict[str, Any],
        original_task: str,
        context: Optional[str],
        attempt: int
    ) -> RepairResult:
        """Attempt to recover from a crash by simplifying the request"""
        
        modifications = []
        
        # Simplify the input if possible
        simplified_state = self._simplify_input(input_state)
        modifications.append("Simplified input state")
        
        # Add error handling instructions
        enhanced_prompt = f"""{original_prompt}

NOTE: A previous attempt encountered an error: {failure_info.error_message}
If you cannot fully complete the task, provide a partial response with what you can determine.
Do NOT raise errors or exceptions."""

        modifications.append("Added error prevention instructions")
        
        try:
            new_output = self._regenerate_with_prompt(enhanced_prompt, simplified_state, attempt)
            
            return RepairResult(
                success=True,
                strategy_used="crash_recovery",
                new_output=new_output,
                modifications_made=modifications
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                reason=f"Crash recovery failed: {str(e)}",
                modifications_made=modifications
            )
    
    def _repair_timeout(
        self,
        failure_info: FailureInfo,
        agent_func: Callable,
        original_prompt: str,
        input_state: Dict[str, Any],
        original_task: str,
        context: Optional[str],
        attempt: int
    ) -> RepairResult:
        """Attempt to recover from timeout by requesting concise output"""
        
        modifications = []
        
        enhanced_prompt = f"""{original_prompt}

IMPORTANT: Be concise. Previous attempt timed out.
Provide a focused, efficient response without unnecessary elaboration.
Prioritize the most important information first."""

        modifications.append("Added conciseness requirements")
        
        try:
            new_output = self._regenerate_with_prompt(enhanced_prompt, input_state, attempt)
            
            return RepairResult(
                success=True,
                strategy_used="timeout_recovery",
                new_output=new_output,
                modifications_made=modifications
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                reason=f"Timeout recovery failed: {str(e)}",
                modifications_made=modifications
            )
    
    def _retrieve_grounding_facts(
        self,
        input_state: Dict[str, Any],
        original_task: str,
        context: Optional[str]
    ) -> str:
        """Extract or retrieve facts for grounding"""
        
        # In a real implementation, this could:
        # - Query a knowledge base
        # - Use RAG to retrieve relevant documents
        # - Extract facts from the input state
        
        # For now, extract facts from input state
        facts = []
        
        for key, value in input_state.items():
            if isinstance(value, str) and len(value) > 20:
                facts.append(f"- {key}: {value[:500]}")
            elif isinstance(value, (int, float, bool)):
                facts.append(f"- {key}: {value}")
        
        if context:
            facts.append(f"- Additional context: {context[:500]}")
        
        return "\n".join(facts) if facts else "No specific facts available. Use only information from the task description."
    
    def _regenerate_with_prompt(
        self,
        prompt: str,
        input_state: Dict[str, Any],
        attempt: int
    ) -> Any:
        """Regenerate output using the LLM"""
        
        # Adjust temperature based on attempt
        if self.repair_config.enable_temperature_adjustment:
            temp = self.repair_config.retry_temperatures[
                min(attempt, len(self.repair_config.retry_temperatures) - 1)
            ]
            # Create new LLM with adjusted temperature
            if self.config.llm_provider == LLMProvider.OPENAI:
                llm = ChatOpenAI(
                    model=self.config.llm_model,
                    temperature=temp,
                    api_key=self.config.openai_api_key
                )
            else:
                llm = ChatAnthropic(
                    model=self.config.llm_model,
                    temperature=temp,
                    api_key=self.config.anthropic_api_key
                )
        else:
            llm = self.llm
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _attempt_format_fix(self, output: Any, error_message: str) -> Optional[Any]:
        """Try to fix format issues directly without regeneration"""
        
        if isinstance(output, str):
            # Try to extract JSON from markdown code blocks
            if "```json" in output:
                try:
                    json_str = output.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                except:
                    pass
            
            # Try to parse as JSON directly
            try:
                return json.loads(output.strip())
            except:
                pass
        
        return None
    
    def _simplify_input(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify input state to reduce complexity"""
        
        simplified = {}
        for key, value in input_state.items():
            if isinstance(value, str) and len(value) > 1000:
                simplified[key] = value[:1000] + "... [truncated]"
            elif isinstance(value, list) and len(value) > 10:
                simplified[key] = value[:10]
            elif isinstance(value, dict) and len(json.dumps(value)) > 1000:
                simplified[key] = {k: v for i, (k, v) in enumerate(value.items()) if i < 5}
            else:
                simplified[key] = value
        
        return simplified
