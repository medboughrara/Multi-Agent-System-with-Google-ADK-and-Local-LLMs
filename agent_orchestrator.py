"""
Agent Orchestrator for coordinating multiple LLM models in a task pipeline.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType:
    CODE_GENERATION = "code-generation"
    CODE_REVIEW = "code-review"
    DOCUMENTATION = "documentation"
    PLANNING = "planning"
    CONVERSATION = "conversation"
    ARCHITECTURE = "architecture"
    DEBUG = "debug"

__all__ = ['AgentOrchestrator', 'TaskType']

class AgentOrchestrator:
    """Coordinates multiple AI models for different tasks in the workflow."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.system_monitor = get_monitor()
        self.task_queue = asyncio.Queue()
        self.results = {}
    
    def _get_model_for_task_type(self, task_type: str) -> str:
        """Maps task types to the most appropriate model."""
        task_model_mapping = {
            TaskType.CODE_GENERATION: ["wizardcoder:7b-python", "codellama:7b"],
            TaskType.CODE_REVIEW: ["codellama:7b", "deepseek-coder-v2:16b"],
            TaskType.DOCUMENTATION: ["neural-chat:7b", "mistral:latest"],
            TaskType.PLANNING: ["mistral:latest", "neural-chat:7b"],
            TaskType.CONVERSATION: ["neural-chat:7b", "mistral:latest"],
            TaskType.ARCHITECTURE: ["deepseek-coder-v2:16b", "codellama:7b"],
            TaskType.DEBUG: ["wizardcoder:7b-python", "codellama:7b"]
        }
        
        models = task_model_mapping.get(task_type, ["mistral:latest"])
        for model in models:
            if model in self.model_manager.MODEL_SPECS:
                return model
        return "mistral:latest"  # fallback to general-purpose model

    async def _run_model(self, model: str, prompt: str) -> str:
        """Run a specific model with the given prompt."""
        try:
            cmd = ["ollama", "run", model, prompt]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Model {model} failed: {stderr.decode()}")
                return ""
                
            return stdout.decode().strip()
        except Exception as e:
            logger.error(f"Error running model {model}: {e}")
            return ""

    async def process_task(self, task_type: str, prompt: str, priority: bool = False) -> str:
        """Process a single task with the most appropriate model."""
        model = self.model_manager.get_best_model_for_task(task_type, priority)
        if not model:
            model = self._get_model_for_task_type(task_type)
            
        logger.info(f"Processing {task_type} task with model: {model}")
        result = await self._run_model(model, prompt)
        return result

    async def process_pipeline(self, tasks: List[Dict[str, str]]) -> Dict[str, str]:
        """Process a pipeline of tasks in the most efficient order."""
        results = {}
        allocations = self.model_manager.get_task_allocation([t["type"] for t in tasks])
        
        # Group tasks by model to minimize model loading/unloading
        tasks_by_model = {}
        for task in tasks:
            model = allocations.get(task["type"]) or self._get_model_for_task_type(task["type"])
            if model not in tasks_by_model:
                tasks_by_model[model] = []
            tasks_by_model[model].append(task)
        
        # Process tasks grouped by model
        for model, model_tasks in tasks_by_model.items():
            model_results = await asyncio.gather(*[
                self.process_task(task["type"], task["prompt"])
                for task in model_tasks
            ])
            
            # Store results
            for task, result in zip(model_tasks, model_results):
                results[task.get("id", task["type"])] = result
        
        return results

    def get_system_status(self) -> Dict:
        """Get current system status including GPU utilization."""
        return {
            "system_info": self.system_monitor.get_system_info(),
            "current_usage": self.system_monitor.get_current_usage(),
            "available_models": [
                {
                    "name": name,
                    "info": self.model_manager.get_model_info(name)
                }
                for name in self.model_manager.MODEL_SPECS.keys()
            ]
        }
