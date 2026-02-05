"""
AEGIS Benchmarks
================
Performance measurement utilities for weather-based replanning.
"""

import time
import csv
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReplanMetrics:
    """Metrics for a single replanning event"""
    trigger: str                    # e.g., "weather_change", "agent_failure"
    latency_ms: float              # Time to compute new plan
    activities_before: int         # Plan size before replanning
    activities_after: int          # Plan size after replanning
    success: bool                  # Whether replanning succeeded
    weather_condition: Optional[str] = None  # The weather that triggered replan
    failed_action: Optional[str] = None      # The action that couldn't execute


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results for multiple replanning scenarios"""
    metrics: List[ReplanMetrics] = field(default_factory=list)

    def add(self, m: ReplanMetrics):
        """Add a replanning metric to the results"""
        self.metrics.append(m)

    def avg_latency(self) -> float:
        """Calculate average replanning latency in milliseconds"""
        if not self.metrics:
            return 0.0
        return sum(m.latency_ms for m in self.metrics) / len(self.metrics)

    def success_rate(self) -> float:
        """Calculate percentage of successful replans (0.0 to 1.0)"""
        if not self.metrics:
            return 0.0
        successes = sum(1 for m in self.metrics if m.success)
        return successes / len(self.metrics)

    def total_replans(self) -> int:
        """Total number of replan attempts"""
        return len(self.metrics)

    def successful_replans(self) -> int:
        """Number of successful replans"""
        return sum(1 for m in self.metrics if m.success)

    def failed_replans(self) -> int:
        """Number of failed replans"""
        return sum(1 for m in self.metrics if not m.success)

    def avg_plan_size_change(self) -> float:
        """Average change in plan size after replanning"""
        successful = [m for m in self.metrics if m.success]
        if not successful:
            return 0.0
        changes = [m.activities_after - m.activities_before for m in successful]
        return sum(changes) / len(changes)

    def min_latency(self) -> float:
        """Minimum replanning latency"""
        if not self.metrics:
            return 0.0
        return min(m.latency_ms for m in self.metrics)

    def max_latency(self) -> float:
        """Maximum replanning latency"""
        if not self.metrics:
            return 0.0
        return max(m.latency_ms for m in self.metrics)

    def to_csv(self, filename: str):
        """Export results to CSV for thesis charts"""
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'trigger', 'latency_ms', 'activities_before',
                'activities_after', 'success', 'weather_condition', 'failed_action'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for m in self.metrics:
                writer.writerow({
                    'trigger': m.trigger,
                    'latency_ms': m.latency_ms,
                    'activities_before': m.activities_before,
                    'activities_after': m.activities_after,
                    'success': m.success,
                    'weather_condition': m.weather_condition or '',
                    'failed_action': m.failed_action or ''
                })

    def summary(self) -> str:
        """Generate a summary string for display"""
        return f"""
Benchmark Results Summary
=========================
Total replans:      {self.total_replans()}
Successful:         {self.successful_replans()}
Failed:             {self.failed_replans()}
Success rate:       {self.success_rate() * 100:.1f}%

Latency (ms):
  Average:          {self.avg_latency():.2f}
  Min:              {self.min_latency():.2f}
  Max:              {self.max_latency():.2f}

Plan size change:   {self.avg_plan_size_change():+.1f} activities (avg)
"""


class ReplanTimer:
    """Context manager for timing replanning operations"""

    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.latency_ms: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        return False


def run_benchmark_scenario(
    controller,
    weather_service,
    initial_state,
    goal,
    weather_changes: List[tuple],
    results: BenchmarkResults
):
    """
    Run a benchmark scenario with weather changes.

    Args:
        controller: ItineraryController instance
        weather_service: MockWeatherService instance
        initial_state: Initial planning state
        goal: Goal state
        weather_changes: List of (location, condition) tuples to inject
        results: BenchmarkResults to record metrics
    """
    # Create initial plan
    if not controller.create_plan(initial_state, goal):
        print("Failed to create initial plan")
        return

    initial_plan_size = sum(len(step) for step in controller.current_plan)

    # Inject weather changes and track replanning
    for location, condition in weather_changes:
        weather_service.inject_weather_change(location, condition)

        with ReplanTimer() as timer:
            success = controller.execute_with_weather_monitoring()

        final_plan_size = sum(len(step) for step in controller.current_plan) if controller.current_plan else 0

        results.add(ReplanMetrics(
            trigger="weather_change",
            latency_ms=timer.latency_ms,
            activities_before=initial_plan_size,
            activities_after=final_plan_size,
            success=success,
            weather_condition=condition.value if hasattr(condition, 'value') else str(condition),
            failed_action=None
        ))


if __name__ == "__main__":
    # Quick demo of the benchmarking utilities
    results = BenchmarkResults()

    # Add some sample metrics for demonstration
    results.add(ReplanMetrics(
        trigger="weather_change",
        latency_ms=45.2,
        activities_before=5,
        activities_after=6,
        success=True,
        weather_condition="rainy"
    ))
    results.add(ReplanMetrics(
        trigger="weather_change",
        latency_ms=52.1,
        activities_before=4,
        activities_after=5,
        success=True,
        weather_condition="stormy"
    ))
    results.add(ReplanMetrics(
        trigger="weather_change",
        latency_ms=38.7,
        activities_before=6,
        activities_after=0,
        success=False,
        weather_condition="stormy"
    ))

    print(results.summary())

    # Export to CSV
    results.to_csv("/tmp/benchmark_demo.csv")
    print("Exported results to /tmp/benchmark_demo.csv")
