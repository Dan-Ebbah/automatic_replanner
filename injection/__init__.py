"""
Failure Injection Framework
===========================
Tools for injecting controlled failures into agent workflows for testing.
"""

from .injector import (
    FailureInjector,
    InjectionConfig,
    InjectionRecord,
    InjectionTrigger,
    FailureMode,
    create_hallucination_injection,
    create_crash_injection,
    create_random_failure
)

__all__ = [
    "FailureInjector",
    "InjectionConfig",
    "InjectionRecord",
    "InjectionTrigger",
    "FailureMode",
    "create_hallucination_injection",
    "create_crash_injection",
    "create_random_failure"
]
