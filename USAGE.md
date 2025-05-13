# Multi-Agent System Usage Guide

## Overview
This system combines multiple AI models to handle different aspects of software development tasks. It includes:
- System resource monitoring
- Multiple specialized AI models
- Both UI and CLI interfaces
- Automatic task distribution and orchestration

## Quick Start

### 1. UI Mode (Recommended)
```bash
python run.py
# or explicitly
python run.py --mode ui
```

This will open a Streamlit interface where you can:
- Monitor system resources (CPU, Memory, GPU) in real-time6

- Input your development tasks or requirements
- Track the progress of different agents
- View and download generated outputs

### 2. CLI Mode
```bash
python run.py --mode cli --prompt "your task description"
```

Example prompts:
- "Create a FastAPI application with user authentication"
- "Build a React component for displaying user profiles"
- "Generate unit tests for my Python code"

## Available Models and Their Specialties

1. WizardCoder (7B)
   - Python coding
   - Debugging
   - Code implementation

2. CodeLlama (7B)
   - General coding
   - Documentation
   - Code review

3. Neural-Chat (7B)
   - Conversation
   - Explanation
   - Documentation

4. Mistral
   - Planning
   - Analysis
   - General tasks

5. DeepSeek Coder (16B)
   - Complex coding
   - Architecture design
   - Code refactoring

## Output Structure

The system generates outputs in the following directories:
- `/outputs/src/` - Generated source code
- `/outputs/tests/` - Generated test files
- `/outputs/agent_logs/` - Detailed agent logs
  - workflow_plan_log.txt - Task breakdown and planning
  - frontend_log.txt - UI/frontend related output
  - backend_log.txt - Server/API related output
  - reviewed_log.txt - Code review and improvements
  - clarified_prompt_log.txt - Clarified requirements

## System Requirements

- Python 3.8 or higher
- NVIDIA GPU with 6GB+ VRAM (for optimal performance)
- Required packages (install via `pip install -r requirements.txt`):
  - torch (with CUDA support)
  - streamlit
  - fastapi
  - gputil
  - psutil

## Best Practices

1. **Task Description**
   - Be as specific as possible
   - Include any technical requirements
   - Specify preferred frameworks or libraries

2. **Resource Management**
   - The system automatically manages model loading based on available resources
   - Larger models (like DeepSeek 16B) require more VRAM
   - Multiple smaller models can run in parallel

3. **Output Management**
   - Check the `/outputs` directory for generated files
   - Review agent logs for detailed reasoning
   - Use the UI to monitor progress in real-time

## Troubleshooting

1. If you encounter GPU memory issues:
   - The system will automatically fall back to smaller models
   - Try closing other GPU-intensive applications
   - Use CLI mode for lower resource usage

2. For best results:
   - Ensure all requirements are installed
   - Keep GPU drivers updated
   - Monitor system resources in the UI
