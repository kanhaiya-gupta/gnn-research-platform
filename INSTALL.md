# Installation Guide

This guide provides step-by-step instructions for installing and setting up the Graph Neural Network Platform.

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package installer
- **Git**: For cloning the repository

## Quick Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd graph-neural-network
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Platform
```bash
# Start frontend (Terminal 1)
python run_frontend.py

# Start backend (Terminal 2)
python run_backend.py
```

### 4. Access the Application
- **Frontend**: http://localhost:5000
- **Backend API**: http://localhost:8001

## Detailed Installation

### Option 1: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv gnn_env

# Activate virtual environment
# On Windows:
gnn_env\Scripts\activate
# On macOS/Linux:
source gnn_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda

```bash
# Create conda environment
conda create -n gnn_platform python=3.9
conda activate gnn_platform

# Install PyTorch and PyTorch Geometric
conda install pytorch torchvision torchaudio -c pytorch
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# Install other dependencies
pip install -r requirements.txt
```

### Option 3: Minimal Installation (Core Features Only)

```bash
# Install only essential packages
pip install torch torch-geometric fastapi uvicorn jinja2
pip install numpy scipy scikit-learn matplotlib
```

## Platform-Specific Installation

### Windows
```bash
# Install Visual C++ Build Tools (if needed)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### macOS
```bash
# Install using Homebrew (optional)
brew install python@3.9

# Install PyTorch
pip install torch torchvision torchaudio
```

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## GPU Support

### NVIDIA GPU (CUDA)
```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with appropriate CUDA version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon (M1/M2)
```bash
# Install PyTorch for Apple Silicon
pip install torch torchvision torchaudio
```

## Verification

### Test Installation
```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test PyTorch Geometric
python -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')"

# Test FastAPI
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
```

### Run Quick Test
```bash
# Test a simple GNN experiment
python -c "
from node_tasks.classification.nodes_classification import get_default_config
config = get_default_config()
config['dataset_name'] = 'synthetic'
config['epochs'] = 5
print('Configuration loaded successfully!')
"
```

## Troubleshooting

### Common Issues

#### 1. PyTorch Geometric Installation Issues
```bash
# Try installing with specific CUDA version
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Or install from source
pip install torch-geometric --no-binary torch-geometric
```

#### 2. CUDA Out of Memory
- Reduce batch size in configuration
- Use smaller models or datasets
- Enable gradient checkpointing

#### 3. Import Errors
```bash
# Reinstall problematic packages
pip uninstall torch-geometric
pip install torch-geometric

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 4. Port Already in Use
```bash
# Kill processes using ports 5000 and 8001
# On Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# On macOS/Linux:
lsof -ti:5000 | xargs kill -9
```

### Performance Optimization

#### GPU Memory Management
```python
# In your configuration
config = {
    'batch_size': 32,  # Reduce if out of memory
    'hidden_channels': 64,  # Reduce for large graphs
    'num_layers': 2,  # Reduce model complexity
}
```

#### CPU Optimization
```python
# Use multiple CPU cores
import torch
torch.set_num_threads(4)  # Adjust based on your CPU
```

## Development Setup

### Install Development Dependencies
```bash
# Install development tools
pip install pytest black flake8 jupyter

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Code Formatting
```bash
# Format code with black
black .

# Check code style with flake8
flake8 .
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_node_classification.py
```

## Docker Installation (Alternative)

### Using Docker
```bash
# Build Docker image
docker build -t gnn-platform .

# Run container
docker run -p 5000:5000 -p 8001:8001 gnn-platform
```

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# Stop services
docker-compose down
```

## Next Steps

After successful installation:

1. **Explore the Dashboard**: Visit http://localhost:5000
2. **Try Quick Examples**: Run the provided code examples
3. **Read Documentation**: Check the `docs/` directory
4. **Run Experiments**: Start with simple tasks like node classification
5. **Customize**: Modify configurations for your specific needs

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the documentation in `docs/`
3. Check GitHub issues for similar problems
4. Create a new issue with detailed error information

## System Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Storage**: 2GB free space
- **CPU**: 2 cores
- **GPU**: Optional (CUDA-compatible for acceleration)

### Recommended Requirements
- **RAM**: 8GB+
- **Storage**: 5GB+ free space
- **CPU**: 4+ cores
- **GPU**: NVIDIA GPU with 4GB+ VRAM 