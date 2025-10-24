#!/bin/bash

# DeepSeek OCR Server Runner
# This script starts the DeepSeek OCR FastAPI server

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}"
}

# Configuration (can be overridden with environment variables)
export MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-OCR}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export BASE_SIZE="${BASE_SIZE:-1024}"
export IMAGE_SIZE="${IMAGE_SIZE:-640}"
export CROP_MODE="${CROP_MODE:-True}"
export MAX_WORKERS="${MAX_WORKERS:-4}"

print_header "DeepSeek OCR Server"
echo ""

# Check if server script exists
if [ ! -f "deepseek_ocr_server.py" ]; then
    print_error "deepseek_ocr_server.py not found!"
    print_error "Please make sure you're running this script from the project root directory."
    exit 1
fi

# Activate environment
print_info "Activating environment..."
if command -v conda &> /dev/null; then
    # Check if conda environment exists
    if conda env list | grep -q "deepseek-ocr"; then
        eval "$(conda shell.bash hook)"
        conda activate deepseek-ocr
        print_info "Conda environment 'deepseek-ocr' activated"
    else
        print_error "Conda environment 'deepseek-ocr' not found!"
        print_error "Please run ./setup.sh first to set up the environment."
        exit 1
    fi
elif [ -d "venv" ]; then
    source venv/bin/activate
    print_info "Virtual environment activated"
else
    print_error "No environment found!"
    print_error "Please run ./setup.sh first to set up the environment."
    exit 1
fi

# Check Python and dependencies
print_info "Checking Python environment..."
python3 -c "import torch, transformers, fastapi" 2>/dev/null || {
    print_error "Required packages not found!"
    print_error "Please run ./setup.sh to install dependencies."
    exit 1
}

# Check CUDA
print_info "Checking CUDA availability..."
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    print_info "CUDA is available: $GPU_NAME"
else
    print_warning "CUDA is not available. The server will run on CPU (very slow)."
    print_warning "For production use, CUDA/GPU is strongly recommended."
fi

# Display configuration
echo ""
print_info "Server Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Base Size: $BASE_SIZE"
echo "  Image Size: $IMAGE_SIZE"
echo "  Crop Mode: $CROP_MODE"
echo "  Max Workers: $MAX_WORKERS"
echo ""

# Create logs directory
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/server_${TIMESTAMP}.log"

print_info "Starting server..."
print_warning "First run will download the model (~6GB) from Hugging Face."
print_warning "This may take several minutes depending on your internet connection."
echo ""
print_info "Server logs will be written to: $LOG_FILE"
print_info "Press Ctrl+C to stop the server"
echo ""
print_header "Server Starting..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    print_info "Shutting down server..."
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start the server with logging
python3 deepseek_ocr_server.py 2>&1 | tee "$LOG_FILE"
