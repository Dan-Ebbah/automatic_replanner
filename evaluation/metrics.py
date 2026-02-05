"""
Metrics Calculator
==================
Calculates evaluation metrics for AEGIS experiments.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import statistics


@dataclass
class ExperimentResult:
    """Result from a single experiment run"""
    
    run_id: str
    success: bool
    
    # Failure info
    failures_injected: int = 0
    failures_detected: int = 0
    
    # Recovery info
    repair_attempts: int = 0
    repair_successes: int = 0
    recompose_attempts: int = 0
    recompose_successes: int = 0
    
    # Timing
    total_time_ms: float = 0.0
    recovery_time_ms: float = 0.0
    
    # Quality (optional)
    output_quality_score: Optional[float] = None
    
    # Metadata
    failure_types: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class MetricsCalculator:
    """
    Calculates aggregate metrics from experiment results.
    """
    
    def __init__(self):
        self.results: List[ExperimentResult] = []
    
    def add_result(self, result: ExperimentResult) -> None:
        """Add an experiment result"""
        self.results.append(result)
    
    def clear(self) -> None:
        """Clear all results"""
        self.results.clear()
    
    def calculate_all(self) -> Dict[str, Any]:
        """Calculate all metrics"""
        
        if not self.results:
            return {"error": "No results to calculate"}
        
        return {
            "summary": self._calculate_summary(),
            "detection": self._calculate_detection_metrics(),
            "recovery": self._calculate_recovery_metrics(),
            "timing": self._calculate_timing_metrics(),
            "quality": self._calculate_quality_metrics(),
            "by_failure_type": self._calculate_by_failure_type()
        }
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        
        n = len(self.results)
        successes = sum(1 for r in self.results if r.success)
        
        return {
            "total_runs": n,
            "successful_runs": successes,
            "failed_runs": n - successes,
            "success_rate": successes / n if n > 0 else 0
        }
    
    def _calculate_detection_metrics(self) -> Dict[str, Any]:
        """Calculate failure detection metrics"""
        
        total_injected = sum(r.failures_injected for r in self.results)
        total_detected = sum(r.failures_detected for r in self.results)
        
        return {
            "total_failures_injected": total_injected,
            "total_failures_detected": total_detected,
            "detection_rate": total_detected / total_injected if total_injected > 0 else 0,
            "false_negative_rate": (total_injected - total_detected) / total_injected if total_injected > 0 else 0
        }
    
    def _calculate_recovery_metrics(self) -> Dict[str, Any]:
        """Calculate recovery metrics"""
        
        total_repair_attempts = sum(r.repair_attempts for r in self.results)
        total_repair_successes = sum(r.repair_successes for r in self.results)
        total_recompose_attempts = sum(r.recompose_attempts for r in self.results)
        total_recompose_successes = sum(r.recompose_successes for r in self.results)
        
        return {
            "repair": {
                "total_attempts": total_repair_attempts,
                "total_successes": total_repair_successes,
                "success_rate": total_repair_successes / total_repair_attempts if total_repair_attempts > 0 else 0
            },
            "recompose": {
                "total_attempts": total_recompose_attempts,
                "total_successes": total_recompose_successes,
                "success_rate": total_recompose_successes / total_recompose_attempts if total_recompose_attempts > 0 else 0
            },
            "overall_recovery_rate": self._calculate_overall_recovery_rate()
        }
    
    def _calculate_overall_recovery_rate(self) -> float:
        """Calculate overall recovery rate"""
        
        total_failures = sum(r.failures_detected for r in self.results)
        total_recovered = sum(
            r.repair_successes + r.recompose_successes
            for r in self.results
        )
        
        return total_recovered / total_failures if total_failures > 0 else 0
    
    def _calculate_timing_metrics(self) -> Dict[str, Any]:
        """Calculate timing metrics"""
        
        total_times = [r.total_time_ms for r in self.results if r.total_time_ms > 0]
        recovery_times = [r.recovery_time_ms for r in self.results if r.recovery_time_ms > 0]
        
        return {
            "total_time": {
                "mean_ms": statistics.mean(total_times) if total_times else 0,
                "median_ms": statistics.median(total_times) if total_times else 0,
                "std_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0,
                "min_ms": min(total_times) if total_times else 0,
                "max_ms": max(total_times) if total_times else 0
            },
            "recovery_time": {
                "mean_ms": statistics.mean(recovery_times) if recovery_times else 0,
                "median_ms": statistics.median(recovery_times) if recovery_times else 0,
                "std_ms": statistics.stdev(recovery_times) if len(recovery_times) > 1 else 0
            }
        }
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate output quality metrics"""
        
        quality_scores = [
            r.output_quality_score
            for r in self.results
            if r.output_quality_score is not None
        ]
        
        if not quality_scores:
            return {"available": False}
        
        return {
            "available": True,
            "mean_score": statistics.mean(quality_scores),
            "median_score": statistics.median(quality_scores),
            "min_score": min(quality_scores),
            "max_score": max(quality_scores)
        }
    
    def _calculate_by_failure_type(self) -> Dict[str, Any]:
        """Calculate metrics grouped by failure type"""
        
        by_type = {}
        
        for result in self.results:
            for failure_type in result.failure_types:
                if failure_type not in by_type:
                    by_type[failure_type] = {
                        "count": 0,
                        "detected": 0,
                        "recovered": 0
                    }
                by_type[failure_type]["count"] += 1
                by_type[failure_type]["detected"] += 1 if result.failures_detected > 0 else 0
                by_type[failure_type]["recovered"] += 1 if result.success else 0
        
        # Calculate rates
        for failure_type, stats in by_type.items():
            stats["detection_rate"] = stats["detected"] / stats["count"] if stats["count"] > 0 else 0
            stats["recovery_rate"] = stats["recovered"] / stats["count"] if stats["count"] > 0 else 0
        
        return by_type
    
    def generate_report(self) -> str:
        """Generate a text report of metrics"""
        
        metrics = self.calculate_all()
        
        lines = [
            "=" * 60,
            "AEGIS EXPERIMENT METRICS REPORT",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"Total runs: {metrics['summary']['total_runs']}",
            f"Success rate: {metrics['summary']['success_rate']:.2%}",
            "",
            "DETECTION",
            "-" * 40,
            f"Failures injected: {metrics['detection']['total_failures_injected']}",
            f"Failures detected: {metrics['detection']['total_failures_detected']}",
            f"Detection rate: {metrics['detection']['detection_rate']:.2%}",
            "",
            "RECOVERY",
            "-" * 40,
            f"Repair success rate: {metrics['recovery']['repair']['success_rate']:.2%}",
            f"Recompose success rate: {metrics['recovery']['recompose']['success_rate']:.2%}",
            f"Overall recovery rate: {metrics['recovery']['overall_recovery_rate']:.2%}",
            "",
            "TIMING",
            "-" * 40,
            f"Mean total time: {metrics['timing']['total_time']['mean_ms']:.2f} ms",
            f"Mean recovery time: {metrics['timing']['recovery_time']['mean_ms']:.2f} ms",
            "",
            "=" * 60
        ]
        
        return "\n".join(lines)
