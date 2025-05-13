"""
Test script to verify model setup and integration.
"""
import asyncio
import logging
from model_interface import ModelInterface
from system_monitor import get_monitor
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_model_setup():
    """Test the model setup and integration."""
    
    # Get system monitor
    monitor = get_monitor()
    
    # Get system information
    print("\nSystem Information:")
    system_info = ModelInterface.get_system_info()
    print(json.dumps(system_info, indent=2))
    
    # Test model availability for each specialization
    print("\nTesting Models by Specialization:")
    for spec in ["orchestrator", "workflow", "frontend", "backend", "reviewer"]:
        primary_model = ModelInterface.get_model_for_specialization(spec)
        fallback_model = ModelInterface.get_model_for_specialization(spec, fallback=True)
        
        print(f"\n{spec.upper()}:")
        print(f"Primary model: {primary_model} (Available: {ModelInterface.is_model_available(primary_model)})")
        print(f"Fallback model: {fallback_model} (Available: {ModelInterface.is_model_available(fallback_model)})")
        
        # Get model config
        config = ModelInterface.get_model_config(primary_model)
        if config:
            print("Configuration:")
            print(json.dumps(config, indent=2))
    
    # Test generation with each available model
    print("\nTesting Generation:")
    test_prompt = "Write a simple hello world function in Python."
    
    for model_name, available in ModelInterface.get_available_models().items():
        if available:
            print(f"\nTesting {model_name}...")
            try:
                response = ModelInterface.generate(model_name, test_prompt)
                print(f"Response: {response[:200]}...")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_model_setup())
