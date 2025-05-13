"""
Model interfaces for both Ollama and local models.
"""
import os
import gc
import json
import psutil
import logging
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from model_config import MODEL_CONFIGS, SPECIALIZATION_MODELS, FALLBACK_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaInterface:
    """Interface for Ollama models."""
    
    @staticmethod
    def list_models() -> List[str]:
        """List all available Ollama models."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                models = [line.split()[0] for line in result.stdout.split('\n')[1:] if line.strip()]
                return models
            return []
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []
    
    @staticmethod
    def is_available(model_name: str) -> bool:
        """Check if a specific model is available in Ollama."""
        return model_name in OllamaInterface.list_models()
    
    @staticmethod
    def generate(model_name: str, prompt: str, **kwargs) -> Tuple[bool, str]:
        """Generate text using an Ollama model."""
        try:
            # Format the command with any model configuration
            config = MODEL_CONFIGS.get(model_name, {})
            cmd = ["ollama", "run"]
              # Set environment variables for Ollama configuration
            env = os.environ.copy()
            if "temperature" in config:
                env["OLLAMA_TEMPERATURE"] = str(config["temperature"])
            if "context_length" in config:
                env["OLLAMA_CONTEXT_LENGTH"] = str(config["context_length"])
            
            # Run the command
            cmd.extend([model_name, prompt])
            
            # Run the model
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                logger.error(f"Ollama generation failed: {result.stderr}")
                return False, result.stderr
        except Exception as e:
            logger.error(f"Error running Ollama model: {e}")
            return False, str(e)

class ModelInterface:
    """Factory class for creating model instances."""
    
    @staticmethod
    def get_model_for_specialization(specialization: str, fallback: bool = False) -> str:
        """Get the appropriate model name for a specialization."""
        if fallback:
            return FALLBACK_MODELS.get(specialization, FALLBACK_MODELS["orchestrator"])
        return SPECIALIZATION_MODELS.get(specialization, SPECIALIZATION_MODELS["orchestrator"])
    
    @staticmethod
    def get_model_config(model_name: str) -> Dict[str, Any]:
        """Get the configuration for a specific model."""
        return MODEL_CONFIGS.get(model_name, {})
    
    @staticmethod
    def get_available_models() -> Dict[str, bool]:
        """Get a list of all configured models and their availability."""
        ollama_models = set(OllamaInterface.list_models())
        return {
            model: model in ollama_models
            for model in set(list(SPECIALIZATION_MODELS.values()) + list(FALLBACK_MODELS.values()))
        }
    
    @staticmethod
    def is_model_available(model_name: str) -> bool:
        """Check if a specific model is available."""
        return OllamaInterface.is_available(model_name)
    
    @staticmethod
    def generate(model_name: str, prompt: str, **kwargs) -> str:
        """Generate text using the specified model."""
        success, result = OllamaInterface.generate(model_name, prompt, **kwargs)
        if success:
            return result
        
        # If Ollama fails, could add fallback to local models here
        raise RuntimeError(f"Failed to generate with model {model_name}: {result}")
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information including available models and resources."""
        memory = psutil.virtual_memory()
        info = {
            "system": {
                "memory": {
                    "total": memory.total / (1024**3),
                    "available": memory.available / (1024**3),
                    "percent": memory.percent
                },
                "cpu": {
                    "count": psutil.cpu_count(),
                    "percent": psutil.cpu_percent()
                }
            },
            "models": ModelInterface.get_available_models(),
            "specializations": {
                spec: {
                    "primary": SPECIALIZATION_MODELS[spec],
                    "fallback": FALLBACK_MODELS[spec]
                }
                for spec in SPECIALIZATION_MODELS.keys()
            }
        }
        return info
