# Multi-Agent System with Google ADK and Local LLMs

A local-first multi-agent system using Google's Agent Development Kit (ADK) for collaborative coding tasks. This system runs entirely on your local machine with models stored locally, without relying on cloud APIs or external hosted services.

## ⚡ Quick Start

```powershell
# Clone the repository
git clone <repository-url>
cd google-A2A

# Install Python dependencies
pip install -r requirements.txt

# Install Ollama (Windows)
winget install ollama

# Download required models
ollama pull neural-chat:7b
ollama pull gemma:7b
ollama pull llama2:latest
ollama pull mistral:latest
ollama pull wizardcoder:7b-python
ollama pull codellama:7b
ollama pull deepseek-coder-v2:16b

# Start the UI
python run.py
# Or use CLI mode
python run.py --mode cli --prompt "your task"
```

## 🎯 Overview

This project implements a modular, local-first agent-based coding assistant where:

- Each agent is powered by a local LLM (e.g., LLaMA, Mistral, or similar) running on your hardware
- Agents collaborate in parallel or sequentially based on task requirements and system performance
- The system outputs a well-structured project folder with code, plans, discussions, and final outputs
- A custom dynamic UI displays real-time conversations, task progress, and allows file/image uploads

## 🔁 Agent Workflow

The system consists of the following specialized agents:

1. **Main Orchestrator Agent**:
   - Interacts with the user
   - Understands the high-level intent
   - Asks follow-up questions to refine the prompt
   - Delegates tasks to specialized agents

2. **Workflow Designer Agent**:
   - Breaks down the task into a sequence or graph of subtasks
   - Assigns subtasks to appropriate agents based on model capabilities
   - Determines if parallel execution is possible based on hardware resources

3. **Coding Agents** (Frontend and Backend):
   - Each specializes in a domain (frontend, backend)
   - Discuss the best architecture, exchange code strategies
   - Iterate collaboratively, each adding improvements

4. **Reviewer Agent**:
   - Reviews the full solution
   - Suggests improvements, checks structure, and performs final QA

## 🛠️ Technical Features

- **Model Wrappers**: Custom ADK-compatible classes that wrap local LLMs
- **Tool Interfaces**: Extended ADK Tool classes to handle file I/O, code execution, and inter-agent messaging
- **Hardware Awareness**: Agents are aware of system performance constraints (VRAM, RAM, CPU load)
- **Parallel Execution**: Tasks can run in parallel where hardware permits
- **File I/O Capabilities**: Agents can read uploaded files or images if supported by the model

## 💻 User Interface

The system includes a dynamic web-based UI built with Streamlit that:

- Shows agent conversations in real-time (chat-style view)
- Displays task progression with visual indicators
- Allows users to upload files (code snippets, diagrams, images)
- Displays system resource usage to help adjust agent behavior dynamically

## 📂 Project Structure

```
google A2A/
├── local_agent_setup_GBT_v.py  # Core agent system implementation
├── multi_agent_ui.py           # Streamlit UI for the agent system
├── my_models.py                # Local model wrapper implementation
├── system_monitor.py           # System resource monitoring
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── outputs/                    # Generated outputs directory
    ├── src/                    # Generated source code
    ├── tests/                  # Generated tests
    ├── agent_logs/             # Agent conversation logs
    ├── README.md               # Project output documentation
    └── architecture.json       # Project architecture details
```

## 🚀 Getting Started

### Prerequisites

#### System Requirements
- Python 3.8 or higher
- NVIDIA GPU with at least 6GB VRAM (recommended)
- 16GB RAM minimum, 32GB recommended
- 50GB free disk space for models
- Windows 10/11 with PowerShell
- CUDA-compatible GPU drivers

#### Required Software
- Python 3.8+
- CUDA Toolkit 12.1 or higher
- Ollama for model management
- Git for version control

#### Required Python Packages
```txt
torch>=2.5.1
torchvision>=0.20.1
streamlit>=1.32.0
fastapi>=0.109.0
gputil>=1.4.0
psutil>=5.9.0
nvidia-ml-py3>=7.352.0
pynvml>=12.0.0
typing-extensions>=4.8.0
python-dotenv>=1.0.0
```

#### Supported Models
The system uses the following models, each specialized for different tasks:

1. **Neural-Chat (7B)**
   - Size: 4.1 GB
   - Use: General conversation, task clarification
   - VRAM Required: ~5GB

2. **WizardCoder (7B-Python)**
   - Size: 3.8 GB
   - Use: Python code generation, debugging
   - VRAM Required: ~4.5GB

3. **CodeLlama (7B)**
   - Size: 3.8 GB
   - Use: Code review, documentation
   - VRAM Required: ~4.5GB

4. **Mistral (Latest)**
   - Size: 4.1 GB
   - Use: Planning, analysis
   - VRAM Required: ~5GB

5. **DeepSeek Coder V2 (16B)**
   - Size: 8.9 GB
   - Use: Complex coding tasks, architecture
   - VRAM Required: ~10GB

### Installation

#### 1. System Setup

1. **Install CUDA and GPU Drivers**:
   ```powershell
   # Check NVIDIA driver version
   nvidia-smi
   
   # If needed, download and install latest drivers from NVIDIA website
   # Download CUDA Toolkit 12.1 from NVIDIA website
   ```

2. **Install Ollama**:
   ```powershell
   # Using winget
   winget install ollama
   
   # Verify installation
   ollama --version
   ```

3. **Clone Repository**:
   ```powershell
   git clone <repository-url>
   cd google-A2A
   ```

4. **Set Up Python Environment**:
   ```powershell
   # Create virtual environment
   python -m venv venv
   .\venv\Scripts\Activate
   
   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Download Required Models**:
   ```powershell
   # Pull all required models
   ollama pull neural-chat:7b
   ollama pull wizardcoder:7b-python
   ollama pull codellama:7b
   ollama pull mistral:latest
   ollama pull deepseek-coder-v2:16b
   
   # Verify models
   ollama list
   ```

6. **Run the System**:
   ```powershell
   # UI Mode (recommended for first use)
   python run.py
   
   # Or CLI Mode
   python run.py --mode cli --prompt "Create a FastAPI application"
   
   # Show detailed usage
   python run.py --help-usage
   ```

### Verification and Testing

1. **Test GPU Setup**:
   ```powershell
   python test_gpu_monitor.py
   ```

2. **Test Model Allocation**:
   ```powershell
   python test_model_allocation.py
   ```

3. **Test Orchestrator**:
   ```powershell
   python test_orchestrator.py
   ```

## 📊 System Monitoring and Resource Management

### Hardware Monitoring
The system includes a comprehensive resource monitoring module that:

- Tracks CPU, RAM, and GPU usage in real-time
- Monitors VRAM allocation and utilization
- Provides GPU temperature and power usage metrics
- Determines optimal model loading strategies

### Resource Management
- **Dynamic Load Balancing**: Automatically adjusts workload based on available resources
- **Smart Model Selection**: Chooses appropriate models based on:
  - Available VRAM
  - Task complexity
  - Required expertise
  - Current system load
- **Parallel Execution Control**: Manages concurrent model execution based on hardware capabilities
- **Memory Management**: 
  - Automatic VRAM cleanup
  - Model unloading when idle
  - Cache management for frequently used models

### Performance Optimization
- **Task Prioritization**: Critical tasks get resource priority
- **Model Sharing**: Efficient memory usage through model sharing when possible
- **Adaptive Batch Sizing**: Adjusts batch sizes based on available resources
- **Pipeline Optimization**: Minimizes model loading/unloading cycles

## 📝 Output Structure and Documentation

### Output Directory Structure
```
outputs/
├── src/                    # Generated source code
│   ├── frontend/          # Frontend components and assets
│   ├── backend/           # Backend services and APIs
│   └── shared/            # Shared utilities and types
├── tests/                 # Test files
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── agent_logs/           # Detailed agent interaction logs
│   ├── orchestrator/     # Main orchestrator logs
│   ├── workflow/         # Workflow designer logs
│   ├── frontend/         # Frontend agent logs
│   ├── backend/         # Backend agent logs
│   └── reviewer/        # Code review logs
├── docs/                # Generated documentation
│   ├── api/            # API documentation
│   ├── architecture/   # Architecture diagrams and docs
│   └── setup/         # Setup and deployment guides
└── metadata/          # Project metadata
    ├── README.md      # Project overview
    ├── ARCHITECTURE.md # Architecture details
    └── metrics.json   # Performance metrics
```

### Documentation Types
1. **Code Documentation**
   - Inline comments and docstrings
   - Type hints and interfaces
   - Usage examples
   - Performance considerations

2. **Architecture Documentation**
   - System design diagrams
   - Component interaction maps
   - Data flow diagrams
   - Deployment architecture

3. **Agent Interaction Logs**
   - Decision-making process
   - Task breakdown and assignments
   - Code review comments
   - Optimization suggestions

4. **Performance Reports**
   - Resource utilization metrics
   - Task completion times
   - Model performance stats
   - Optimization opportunities

## 🔄 Workflow Execution and Task Management

### Execution Modes

1. **Parallel Execution**
   - Multiple agents work simultaneously
   - Automatic resource allocation
   - Load balancing between models
   - Real-time progress monitoring

2. **Sequential Execution**
   - Step-by-step task processing
   - Resource-efficient operation
   - Detailed progress tracking
   - Guaranteed task ordering

3. **Hybrid Execution**
   - Dynamic switching between modes
   - Resource-aware scheduling
   - Priority-based execution
   - Optimal task grouping

### Task Management Features

1. **Task Prioritization**
   - Critical path analysis
   - Resource requirement assessment
   - Dependency resolution
   - Priority queue management

2. **Resource Optimization**
   - Smart model loading
   - Memory management
   - GPU utilization balancing
   - Cache optimization

3. **Error Handling**
   - Automatic retry mechanism
   - Fallback strategies
   - Error recovery
   - State preservation

### Performance Monitoring

1. **Real-time Metrics**
   - Task completion rates
   - Resource utilization
   - Model performance
   - System health

2. **Performance Analysis**
   - Bottleneck detection
   - Resource usage patterns
   - Optimization opportunities
   - Historical performance

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Code Contributions**
   - Fork the repository
   - Create a feature branch
   - Submit a pull request

2. **Documentation**
   - Improve installation guides
   - Add usage examples
   - Update model information

3. **Testing**
   - Add test cases
   - Report bugs
   - Suggest improvements

4. **Model Support**
   - Add new model integrations
   - Optimize existing models
   - Update model configurations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Support and Contact

- GitHub Issues: [Create an issue](https://github.com/yourusername/google-A2A/issues)
- Documentation: See the `docs` folder
- Questions: Check the discussions section