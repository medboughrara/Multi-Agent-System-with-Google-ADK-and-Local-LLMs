#!/usr/bin/env python3
"""
Test script for the Multi-Agent System with Google ADK and Local LLMs.
This script runs a simple test to verify that all components are working correctly.
"""

import os
import sys
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
try:
    from my_models import LocalModelWrapper
    from system_monitor import get_monitor
    from local_agent_setup_GBT_v import run_agent_system
    logger.info("Successfully imported system components")
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    sys.exit(1)

def test_model_wrapper():
    """Test the LocalModelWrapper class."""
    logger.info("Testing LocalModelWrapper...")
    
    try:
        model = LocalModelWrapper("test-model")
        logger.info("Created model wrapper instance")
        
        response = model.generate("Hello, world!")
        logger.info(f"Model generated response: {response[:50]}...")
        
        model_info = model.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing model wrapper: {e}")
        return False

def test_system_monitor():
    """Test the SystemMonitor class."""
    logger.info("Testing SystemMonitor...")
    
    try:
        monitor = get_monitor()
        logger.info("Got system monitor instance")
        
        usage = monitor.get_current_usage()
        logger.info(f"Current usage: CPU {usage['cpu']:.1f}%, Memory {usage['memory']:.1f}%")
        
        system_info = monitor.get_system_info()
        logger.info(f"System info: {system_info}")
        
        can_run_parallel = monitor.can_run_parallel(2)
        logger.info(f"Can run 2 models in parallel: {can_run_parallel}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing system monitor: {e}")
        return False

def test_agent_system():
    """Test the agent system with a simple prompt."""
    logger.info("Testing agent system...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Run a simple test prompt
        test_prompt = "Create a simple calculator web app"
        logger.info(f"Running agent system with prompt: {test_prompt}")
        
        start_time = time.time()
        result = run_agent_system(test_prompt)
        end_time = time.time()
        
        logger.info(f"Agent system completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Result: {result[:100]}...")
        
        # Check if output files were created
        if os.path.exists("outputs/README.md") and os.path.exists("outputs/architecture.json"):
            logger.info("Output files were created successfully")
            return True
        else:
            logger.error("Output files were not created")
            return False
    except Exception as e:
        logger.error(f"Error testing agent system: {e}")
        return False

def main():
    """Run all tests and report results."""
    print("=== Testing Multi-Agent System ===\n")
    
    tests = {
        "Model Wrapper": test_model_wrapper,
        "System Monitor": test_system_monitor,
        "Agent System": test_agent_system
    }
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests.items():
        print(f"Running test: {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
            if not result:
                all_passed = False
            print(f"Test {test_name}: {'PASSED' if result else 'FAILED'}\n")
        except Exception as e:
            logger.error(f"Exception in test {test_name}: {e}")
            results[test_name] = False
            all_passed = False
            print(f"Test {test_name}: FAILED (exception)\n")
    
    # Print summary
    print("\n=== Test Summary ===")
    for test_name, result in results.items():
        print(f"{test_name}: {'PASSED' if result else 'FAILED'}")
    
    print(f"\nOverall result: {'PASSED' if all_passed else 'FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())