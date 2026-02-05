"""
Results Collector
=================
Collects and stores experiment results for analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .metrics import ExperimentResult, MetricsCalculator


class ResultsCollector:
    """
    Collects experiment results and saves them to disk.
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment: Optional[str] = None
        self.results: List[ExperimentResult] = []
        self.metadata: Dict[str, Any] = {}
    
    def start_experiment(
        self,
        name: str,
        description: str = "",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start a new experiment"""
        
        self.current_experiment = name
        self.results = []
        self.metadata = {
            "name": name,
            "description": description,
            "config": config or {},
            "started_at": datetime.now().isoformat(),
            "completed_at": None
        }
    
    def add_result(self, result: ExperimentResult) -> None:
        """Add a result to the current experiment"""
        self.results.append(result)
    
    def end_experiment(self) -> Dict[str, Any]:
        """End the current experiment and save results"""
        
        self.metadata["completed_at"] = datetime.now().isoformat()
        
        # Calculate metrics
        calculator = MetricsCalculator()
        for result in self.results:
            calculator.add_result(result)
        metrics = calculator.calculate_all()
        
        # Prepare output
        output = {
            "metadata": self.metadata,
            "metrics": metrics,
            "results": [self._result_to_dict(r) for r in self.results]
        }
        
        # Save to file
        filename = f"{self.current_experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"Results saved to: {filepath}")
        
        return output
    
    def _result_to_dict(self, result: ExperimentResult) -> Dict[str, Any]:
        """Convert ExperimentResult to dictionary"""
        return asdict(result)
    
    def load_experiment(self, filepath: str) -> Dict[str, Any]:
        """Load a previous experiment from file"""
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_experiments(self) -> List[str]:
        """List all saved experiments"""
        
        files = list(self.output_dir.glob("*.json"))
        return [f.name for f in sorted(files)]
    
    def compare_experiments(
        self,
        filepaths: List[str]
    ) -> Dict[str, Any]:
        """Compare metrics across multiple experiments"""
        
        comparison = {}
        
        for filepath in filepaths:
            exp = self.load_experiment(filepath)
            name = exp["metadata"]["name"]
            comparison[name] = exp["metrics"]["summary"]
        
        return comparison
