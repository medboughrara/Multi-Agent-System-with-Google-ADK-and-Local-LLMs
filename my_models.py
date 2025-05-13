"""
Local model wrapper for Google ADK to use with locally hosted LLMs.
This module provides a wrapper class that allows local LLMs to be used with Google's Agent Development Kit.
"""
import os
import gc
import psutil
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    logger.info(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    DEVICE = torch.device("cuda")
    # Configure GPU memory settings
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for better performance
    torch.backends.cudnn.benchmark = True  # Enable cudnn benchmarking
    # Enable memory efficient attention if available
    try:
        import xformers
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        USE_XFORMERS = True
        logger.info("xformers memory efficient attention enabled")
    except ImportError:
        USE_XFORMERS = False
        logger.info("xformers not available, using standard attention")
else:
    logger.warning("CUDA is not available. Using CPU.")
    DEVICE = torch.device("cpu")
    USE_XFORMERS = False

class LocalModelWrapper:
    """
    A wrapper for local LLMs to be used with Google's Agent Development Kit.
    This class provides an interface to use local models with hardware acceleration.
    """
    
    def __init__(self, model_name: str, model_params: Optional[Dict[str, Any]] = None):
        """Initialize the local model wrapper."""
        self.model_name = model_name
        self.model_params = model_params or {}
        self.is_initialized = False
        self.model = None
        self.tokenizer = None
        self.device = DEVICE
        self.generation_config = None
        
        # Set optimized model parameters for GPU
        if CUDA_AVAILABLE:
            self.model_params.update({
                'torch_dtype': torch.float16,  # Use half precision
                'low_cpu_mem_usage': True,
                'device_map': 'auto'  # Let transformers handle device placement
            })
        
        # Log available system memory
        memory = psutil.virtual_memory()
        logger.info(f"Available system memory: {memory.available / (1024**3):.2f} GB")
        
        self._initialize_model()
    
    def _set_generation_config(self):
        """Set up generation configuration."""
        self.generation_config = GenerationConfig(
            max_length=2048,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            do_sample=True,
            pad_token_id=None,
            eos_token_id=None
        )
    
    def _initialize_model(self):
        """Initialize the model based on type."""
        if self.model_name == "test-model":
            # Simplified test model
            self.model = "test-model"
            self.is_initialized = True
            return
            
        try:
            # Check if model is available in Ollama
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                models = [line.split()[0] for line in result.stdout.split('\n')[1:] if line.strip()]
                if self.model_name in models:
                    logger.info(f"Using Ollama model: {self.model_name}")
                    self.is_initialized = True
                    return
                    
            logger.warning(f"Model {self.model_name} not found in Ollama, falling back to local model")
            self._initialize_local_model()
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
        
        if self.model:
            self.is_initialized = True
            if CUDA_AVAILABLE:
                if USE_XFORMERS:
                    self.model.enable_xformers_memory_efficient_attention()
                self.model.cuda()
                # Enable model specific optimizations
                if hasattr(self.model, "half"):
                    self.model = self.model.half()  # Use FP16 if available
                torch.cuda.empty_cache()
    
    def _load_model_and_tokenizer(self, model_id: str, use_cache: bool = True):
        """Load model and tokenizer with appropriate settings."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if CUDA_AVAILABLE else torch.float32,
                low_cpu_mem_usage=True,
                use_cache=use_cache,
                device_map="auto" if CUDA_AVAILABLE else None
            )
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None, None
    
    def _load_main_model(self):
        """Load the main orchestrator model."""
        logger.info(f"Initialized LocalModelWrapper for model: {self.model_name} on {self.device}")
        self.model, self.tokenizer = self._load_model_and_tokenizer("meta-llama/Llama-2-7b-chat-hf")
    
    def _load_workflow_model(self):
        """Load the workflow designer model."""
        logger.info(f"Initialized LocalModelWrapper for model: {self.model_name} on {self.device}")
        self.model, self.tokenizer = self._load_model_and_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
    
    def _load_frontend_model(self):
        """Load the frontend specialist model."""
        logger.info(f"Initialized LocalModelWrapper for model: {self.model_name} on {self.device}")
        self.model, self.tokenizer = self._load_model_and_tokenizer("codellama/CodeLlama-34b-Instruct-hf")
    
    def _load_backend_model(self):
        """Load the backend specialist model."""
        logger.info(f"Initialized LocalModelWrapper for model: {self.model_name} on {self.device}")
        self.model, self.tokenizer = self._load_model_and_tokenizer("codellama/CodeLlama-34b-Instruct-hf")
    
    def _load_reviewer_model(self):
        """Load the code reviewer model."""
        logger.info(f"Initialized LocalModelWrapper for model: {self.model_name} on {self.device}")
        self.model, self.tokenizer = self._load_model_and_tokenizer("microsoft/codebert-base")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model."""
        if not self.is_initialized:
            self._initialize_model()
        
        if self.model_name == "test-model":
            return self._simulate_response(prompt)
            
        try:
            logger.info(f"Generating response with {self.model_name} for prompt: {prompt[:50]}...")
            
            # Use Ollama for generation if available
            try:
                import subprocess
                cmd = ["ollama", "run", self.model_name, prompt]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception as e:
                logger.warning(f"Ollama generation failed, falling back to local model: {e}")
            
            # Fallback to local model if Ollama fails
            if not self.tokenizer:
                raise RuntimeError("Tokenizer not initialized. Please ensure model is properly loaded.")
                
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            if CUDA_AVAILABLE:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Update generation config with any provided kwargs
            gen_config = self.generation_config.copy()
            for k, v in kwargs.items():
                setattr(gen_config, k, v)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clear GPU memory if needed
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"Model {self.model_name} initialized successfully on {self.device}")
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _simulate_response(self, prompt: str) -> str:
        """Simulate a response for testing."""
        return f"Hello, world! ({prompt})"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        memory = psutil.virtual_memory()
        
        info = {
            'name': self.model_name,
            'parameters': self.model_params,
            'is_initialized': self.is_initialized,
            'type': 'local',
            'status': 'ready',
            'memory_usage': {
                'total': memory.total / (1024**3),
                'available': memory.available / (1024**3)
            }
        }
        
        if CUDA_AVAILABLE:
            info.update({
                'gpu_available': True,
                'cuda_version': torch.version.cuda,
                'gpu_device': torch.cuda.get_device_name(0),
                'gpu_memory': {
                    'allocated': torch.cuda.memory_allocated(0) / (1024**3),
                    'reserved': torch.cuda.memory_reserved(0) / (1024**3)
                }
            })
        
        return info
    
    def __del__(self):
        """Cleanup when the wrapper is destroyed."""
        if self.model_name != "test-model":
            logger.info(f"Cleaning up resources for model: {self.model_name}")
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                gc.collect()