

# Specialization to model mapping
SPECIALIZATION_MODELS = {
    "orchestrator": "neural-chat:7b",  # Good at conversation and task understanding
    "workflow": "mistral:latest",      # Strong at planning and analysis
    "frontend": "codellama:7b",        # Specialized in frontend development
    "backend": "wizardcoder:7b-python", # Python specialist
    "reviewer": "deepseek-coder-v2:16b" # Large model for complex code review
}

# Fallback models (smaller versions) when resources are limited
FALLBACK_MODELS = {
    "orchestrator": "mistral:latest",
    "workflow": "neural-chat:7b",
    "frontend": "wizardcoder:7b-python",
    "backend": "codellama:7b",
    "reviewer": "codellama:7b"
}

# Model configurations
MODEL_CONFIGS = {
    "neural-chat:7b": {
        "context_length": 4096,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.1
    },
    "mistral:latest": {
        "context_length": 8192,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.2
    },
    "wizardcoder:7b-python": {
        "context_length": 4096,
        "temperature": 0.5,
        "top_p": 0.95,
        "repetition_penalty": 1.1
    },
    "codellama:7b": {
        "context_length": 4096,
        "temperature": 0.6,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    },
    "deepseek-coder-v2:16b": {
        "context_length": 16384,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.2
    }
}
