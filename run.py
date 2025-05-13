#!/usr/bin/env python3
"""
Run script for the Multi-Agent System with Google ADK and Local LLMs.
This script provides a simple command-line interface to run either the UI or CLI version.
"""

import os
import sys
import argparse
import subprocess

def show_usage():
    """Display usage information."""
    try:
        with open("USAGE.md", "r") as f:
            print(f.read())
    except FileNotFoundError:
        print("For detailed usage instructions, please refer to USAGE.md")

def main():
    """Main entry point for the run script."""
    parser = argparse.ArgumentParser(
        description="Run the Multi-Agent System with Google ADK and Local LLMs"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["ui", "cli", "verify"], 
        default="ui",
        help="Run mode: 'ui' for Streamlit interface, 'cli' for command line, 'verify' for system verification (default: ui)"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str,
        help="Initial prompt for CLI mode (optional)"
    )
    
    parser.add_argument(
        "--help-usage",
        action="store_true",
        help="Show detailed usage information"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Specify a model to use (optional, defaults to specialization-appropriate model)"
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
      # Run the selected mode
    if args.help_usage:
        show_usage()
        return

    if args.mode == "verify":
        print("Verifying system setup...")
        subprocess.run(["python", "test_model_setup.py"])
        return
        
    if args.mode == "ui":
        print("Starting Streamlit UI...")
        if args.model:
            os.environ["DEFAULT_MODEL"] = args.model
        subprocess.run(["streamlit", "run", "multi_agent_ui.py"])
    else:
        print("Running in CLI mode...")
        if args.prompt:
            os.environ["INITIAL_PROMPT"] = args.prompt
        if args.model:
            os.environ["DEFAULT_MODEL"] = args.model
        subprocess.run(["python", "local_agent_setup_GBT_v.py"])

if __name__ == "__main__":
    main()