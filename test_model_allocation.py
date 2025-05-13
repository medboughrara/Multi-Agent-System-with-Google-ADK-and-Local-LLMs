"""Test the model manager with various tasks."""
from model_manager import ModelManager
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_allocation():
    # Initialize system monitoring
    from system_monitor import SystemMonitor
    system_monitor = SystemMonitor()
    system_monitor.start_monitoring()
    
    manager = ModelManager()
    
    # Test single task allocations
    tasks = [
        "python",
        "coding",
        "documentation",
        "complex-coding",
        "conversation"
    ]
    
    print("\nTesting individual task allocations:")
    for task in tasks:
        model = manager.get_best_model_for_task(task)
        print(f"Task: {task:15} -> Model: {model if model else 'No suitable model'}")
    
    # Test multi-task allocation
    print("\nTesting multi-task allocation:")
    multi_tasks = ["coding", "documentation", "conversation"]
    allocations = manager.get_task_allocation(multi_tasks)
    print("Task Allocations:")
    print(json.dumps(allocations, indent=2))
    
    # Print detailed model information
    print("\nDetailed Model Information:")
    for model_name in manager.MODEL_SPECS.keys():
        info = manager.get_model_info(model_name)
        print(f"\nModel: {model_name}")
        print(json.dumps(info, indent=2))

if __name__ == "__main__":
    test_model_allocation()
