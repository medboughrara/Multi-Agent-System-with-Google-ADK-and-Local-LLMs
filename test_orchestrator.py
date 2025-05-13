"""Test the agent orchestrator with a sample workflow."""
import asyncio
import logging
from agent_orchestrator import AgentOrchestrator, TaskType
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

async def test_workflow():
    orchestrator = AgentOrchestrator()
    
    # Print initial system status
    print("\nInitial System Status:")
    print(json.dumps(orchestrator.get_system_status(), indent=2))
    
    # Define a sample workflow
    workflow = [
        {
            "id": "task1",
            "type": TaskType.PLANNING,
            "prompt": "Create a plan for implementing a REST API with FastAPI"
        },
        {
            "id": "task2",
            "type": TaskType.CODE_GENERATION,
            "prompt": "Write a basic FastAPI application with a hello world endpoint"
        },
        {
            "id": "task3",
            "type": TaskType.DOCUMENTATION,
            "prompt": "Generate documentation for the FastAPI hello world application"
        }
    ]
    
    print("\nProcessing workflow...")
    results = await orchestrator.process_pipeline(workflow)
    
    print("\nWorkflow Results:")
    for task_id, result in results.items():
        print(f"\n=== {task_id} ===")
        print(result)
    
    # Print final system status
    print("\nFinal System Status:")
    print(json.dumps(orchestrator.get_system_status(), indent=2))

if __name__ == "__main__":
    asyncio.run(test_workflow())
