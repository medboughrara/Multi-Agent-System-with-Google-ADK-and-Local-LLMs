"""
Model Manager for coordinating different LLM models based on tasks and system resources.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from system_monitor import get_monitor, SystemMonitor
import subprocess
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSpec:
    def __init__(self, name: str, size_gb: float, specialties: List[str], min_vram_gb: float):
        self.name = name
        self.size_gb = size_gb
        self.specialties = specialties
        self.min_vram_gb = min_vram_gb

class ModelManager:
    """Manages multiple LLM models and their resource requirements."""
    
    # Model specifications with their specialties and requirements
    MODEL_SPECS = {
        "wizardcoder:7b-python": ModelSpec(
            "wizardcoder:7b-python",
            3.8,
            ["python", "coding", "debugging"],
            4.5  # Required VRAM including overhead
        ),
        "codellama:7b": ModelSpec(
            "codellama:7b",
            3.8,
            ["coding", "documentation", "code-review"],
            4.5
        ),
        "deepseek-coder-v2:16b": ModelSpec(
            "deepseek-coder-v2:16b",
            8.9,
            ["complex-coding", "architecture", "refactoring"],
            10.0
        ),
        "neural-chat:7b": ModelSpec(
            "neural-chat:7b",
            4.1,
            ["conversation", "explanation", "documentation"],
            5.0
        ),
        "mistral:latest": ModelSpec(
            "mistral:latest",
            4.1,
            ["general", "analysis", "planning"],
            5.0
        )
    }

    def __init__(self):
        """Initialize the model manager."""
        self.system_monitor = get_monitor()
        self.active_models: Dict[str, bool] = {}
        self._verify_models()

    def _verify_models(self):
        """Verify which models are actually available locally."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            available_models = [line.split()[0] for line in result.stdout.split('\n')[1:] if line.strip()]
            
            # Update MODEL_SPECS to only include available models
            self.MODEL_SPECS = {k: v for k, v in self.MODEL_SPECS.items() 
                              if k in available_models}
            
            logger.info(f"Available models: {list(self.MODEL_SPECS.keys())}")
        except Exception as e:
            logger.error(f"Error verifying models: {e}")

    def get_best_model_for_task(self, task_type: str, priority: bool = False) -> Optional[str]:
        """
        Select the best model for a given task type.
        
        Args:
            task_type: Type of task (e.g., 'coding', 'documentation', etc.)
            priority: If True, will try to allocate resources even if system is under load
            
        Returns:
            Model name or None if no suitable model is available
        """
        suitable_models = []
        for model_name, spec in self.MODEL_SPECS.items():
            if task_type in spec.specialties:
                if self.can_run_model(spec, priority):
                    suitable_models.append((model_name, spec))
        
        if not suitable_models:
            return None
            
        # Sort by size (prefer smaller models if they can handle the task)
        suitable_models.sort(key=lambda x: x[1].size_gb)
        return suitable_models[0][0]

    def can_run_model(self, model_spec: ModelSpec, priority: bool = False) -> bool:
        """Check if the system can run a specific model."""
        gpu_info = self.system_monitor.get_current_usage()
        
        # Get current GPU memory usage
        gpu_memory_used = gpu_info.get('gpu_memory', 0)
        
        # If priority is False, ensure we keep some GPU memory free
        memory_threshold = 90 if priority else 75
        
        if gpu_memory_used > memory_threshold:
            return False
            
        # Check if we have enough VRAM
        return self.system_monitor.can_run_parallel(1, model_spec.min_vram_gb)

    def get_task_allocation(self, tasks: List[str]) -> Dict[str, str]:
        """
        Allocate models to multiple tasks optimally.
        
        Args:
            tasks: List of task types
            
        Returns:
            Dictionary mapping tasks to model names
        """
        allocations = {}
        priority_pass = False
        
        for task in tasks:
            model = self.get_best_model_for_task(task, priority_pass)
            if model:
                allocations[task] = model
        
        # Try again with priority for unallocated tasks
        unallocated = [t for t in tasks if t not in allocations]
        if unallocated:
            priority_pass = True
            for task in unallocated:
                model = self.get_best_model_for_task(task, priority_pass)
                if model:
                    allocations[task] = model
        
        return allocations

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model."""
        if model_name not in self.MODEL_SPECS:
            return None
            
        spec = self.MODEL_SPECS[model_name]
        return {
            "name": spec.name,
            "size_gb": spec.size_gb,
            "specialties": spec.specialties,
            "min_vram_gb": spec.min_vram_gb,
            "can_run": self.can_run_model(spec)
        }
