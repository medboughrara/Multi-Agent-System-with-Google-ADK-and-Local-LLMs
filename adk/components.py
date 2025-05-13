"""
Core components for the ADK system.
Provides base classes for Agents, Tools, and workflow control structures.
"""

import logging
from typing import List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .runtime import Context

logger = logging.getLogger(__name__)

class Agent:
    """Base class for all agents in the system."""
    
    def __init__(self, model):
        """Initialize the agent with a model."""
        self.model = model
        self.name = "base_agent"
    
    def run(self, input_data: str, context: Context) -> str:
        """Run the agent's processing logic."""
        raise NotImplementedError("Agents must implement run()")

class Tool:
    """Base class for all tools that agents can use."""
    
    def run(self, input_data: str, context: Context) -> str:
        """Run the tool's functionality."""
        raise NotImplementedError("Tools must implement run()")

class Sequential:
    """Execute a sequence of components one after another."""
    
    def __init__(self, components: List[Any]):
        """Initialize with a list of components to execute sequentially."""
        self.components = components
    
    def run(self, input_data: str, context: Context) -> str:
        """Run all components in sequence, passing output as input to the next."""
        current_output = input_data
        try:
            for component in self.components:
                logger.info(f"Running sequential component: {component.__class__.__name__}")
                current_output = component.run(current_output, context)
            return current_output
        except Exception as e:
            logger.error(f"Error in sequential execution: {e}")
            raise

class Parallel:
    """Execute components in parallel using thread pool."""
    
    def __init__(self, components: List[Any], max_workers: Optional[int] = None):
        """Initialize with a list of components to execute in parallel."""
        self.components = components
        self.max_workers = max_workers or len(components)
    
    def run(self, input_data: str, context: Context) -> str:
        """Run all components in parallel and combine their outputs."""
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(component.run, input_data, context)
                    for component in self.components
                ]
                
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in parallel component: {e}")
                        raise
                
                # Combine results
                return "\n".join(results)
        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")
            raise

class Loop:
    """Execute a component repeatedly until a condition is met."""
    
    def __init__(self, component: Any, max_iterations: int = 10, 
                 condition_fn=lambda output, context: False):
        """
        Initialize loop component.
        
        Args:
            component: Component to execute repeatedly
            max_iterations: Maximum number of iterations
            condition_fn: Function that takes (output, context) and returns True when loop should stop
        """
        self.component = component
        self.max_iterations = max_iterations
        self.condition_fn = condition_fn
    
    def run(self, input_data: str, context: Context) -> str:
        """Run the component repeatedly until condition is met or max iterations reached."""
        current_output = input_data
        iteration = 0
        
        try:
            while iteration < self.max_iterations:
                logger.info(f"Loop iteration {iteration + 1}/{self.max_iterations}")
                
                current_output = self.component.run(current_output, context)
                iteration += 1
                
                if self.condition_fn(current_output, context):
                    logger.info("Loop condition met, stopping iterations")
                    break
            
            return current_output
        except Exception as e:
            logger.error(f"Error in loop execution: {e}")
            raise
