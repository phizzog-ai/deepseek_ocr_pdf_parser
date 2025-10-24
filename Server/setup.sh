#!/bin/bash

# DeepSeek OCR Setup Script
# This script sets up the environment and installs all dependencies for the DeepSeek OCR server

set -e  # Exit on error

echo "========================================="
echo "DeepSeek OCR Server Setup"
echo "========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on the correct server
print_info "Checking environment..."
if [ -f "/etc/lsb-release" ]; then
    source /etc/lsb-release
    print_info "OS: $DISTRIB_DESCRIPTION"
else
    print_warning "Could not detect OS version"
fi

# Check CUDA availability
print_info "Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    print_error "nvidia-smi not found. CUDA is required for this project."
    print_error "Please ensure NVIDIA drivers and CUDA are properly installed."
    exit 1
fi

# Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"

REQUIRED_VERSION="3.12.0"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python 3.12+ is required. Current version: $PYTHON_VERSION"
    print_info "Please install Python 3.12 or later."
    exit 1
fi

# Check if conda is available
if command -v conda &> /dev/null; then
    print_info "Conda detected: $(conda --version)"
    USE_CONDA=true
else
    print_warning "Conda not found. Will use venv instead."
    USE_CONDA=false
fi

# Set up environment
if [ "$USE_CONDA" = true ]; then
    print_info "Setting up Conda environment: deepseek-ocr"

    # Check if environment already exists
    if conda env list | grep -q "deepseek-ocr"; then
        print_warning "Conda environment 'deepseek-ocr' already exists."
        read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            conda env remove -n deepseek-ocr -y
        else
            print_info "Using existing environment."
        fi
    fi

    if ! conda env list | grep -q "deepseek-ocr"; then
        print_info "Creating new Conda environment..."
        conda create -n deepseek-ocr python=3.12 -y
    fi

    print_info "Activating Conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate deepseek-ocr
else
    print_info "Setting up virtual environment: venv"

    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists."
        read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf venv
        fi
    fi

    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi

    print_info "Activating virtual environment..."
    source venv/bin/activate
fi

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
print_info "Installing PyTorch 2.6.0 with CUDA 11.8 support..."
print_warning "This may take several minutes..."
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and related packages
print_info "Installing transformers and tokenizers..."
pip install transformers==4.46.3 tokenizers==0.20.3

# Install flash-attention (this may take a while to compile)
print_info "Installing flash-attention 2.7.3..."
print_warning "This will compile from source and may take 10-20 minutes..."
pip install flash-attn==2.7.3 --no-build-isolation

# Install FastAPI and uvicorn
print_info "Installing FastAPI and uvicorn..."
pip install fastapi uvicorn[standard]

# Install additional dependencies
print_info "Installing additional dependencies..."
pip install pydantic pillow python-multipart aiofiles

# Create requirements.txt for future reference
print_info "Creating requirements.txt..."
pip freeze > requirements.txt

# Verify installation
print_info "Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python3 -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

# Test flash-attention import
print_info "Testing flash-attention import..."
if python3 -c "import flash_attn" 2>/dev/null; then
    print_info "flash-attention successfully installed"
else
    print_warning "flash-attention import failed. The model may fall back to standard attention."
fi

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p logs
mkdir -p outputs

echo ""
echo "========================================="
print_info "Setup completed successfully!"
echo "========================================="
echo ""

if [ "$USE_CONDA" = true ]; then
    print_info "To activate the environment, run:"
    echo "    conda activate deepseek-ocr"
else
    print_info "To activate the environment, run:"
    echo "    source venv/bin/activate"
fi

echo ""
print_info "To start the server, run:"
echo "    ./run_server.sh"
echo ""

print_warning "Note: On first run, the model will be downloaded from Hugging Face."
print_warning "This may take several minutes depending on your internet connection."
print_warning "The model is approximately 6GB in size."
echo ""
