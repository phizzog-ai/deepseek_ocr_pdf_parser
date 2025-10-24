# DeepSeek OCR Server

A production-ready FastAPI server for optical character recognition using the [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) model. This server provides a simple REST API for extracting text from images, converting documents to markdown, parsing figures, and locating text within images.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Testing](#testing)
- [Performance](#performance)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [What I Learned](#what-i-learned)
- [License](#license)

## Overview

This project implements a FastAPI-based server that wraps the DeepSeek-OCR model, providing a simple HTTP API for various OCR tasks. The server is designed to run on GPU-enabled Ubuntu servers and can handle multiple concurrent requests efficiently.

**Key Capabilities:**
- General OCR text extraction
- Document-to-markdown conversion
- Figure and diagram parsing
- Text localization and grounding
- Multiple image format support
- Remote API access

## Features

### 🚀 Production Ready
- **FastAPI Framework**: Modern, fast async web framework
- **GPU Acceleration**: Optimized for NVIDIA CUDA GPUs
- **Async Processing**: Non-blocking request handling with ThreadPoolExecutor
- **Health Checks**: Built-in health monitoring endpoints
- **Logging**: Comprehensive logging for debugging and monitoring

### 🎯 Multiple OCR Tasks
- **Free OCR**: Extract all text from images
- **Document to Markdown**: Convert document images to structured markdown
- **Figure Parsing**: Parse diagrams, charts, and figures
- **Text Localization**: Find specific text locations in images

### 🛠️ Developer Friendly
- **Easy Setup**: Automated setup script handles all dependencies
- **Test Client**: Included Python client for testing
- **Docker Ready**: Can be containerized for easy deployment
- **Configuration**: Environment-based configuration
- **Documentation**: Comprehensive API documentation

## Architecture

The server consists of several key components:

```
┌─────────────────────────────────────────────┐
│           Client Application                │
│  (test_client.py or any HTTP client)        │
└──────────────────┬──────────────────────────┘
                   │ HTTP/JSON
                   ▼
┌─────────────────────────────────────────────┐
│         FastAPI Server (Port 8000)          │
│  ┌───────────────────────────────────────┐  │
│  │  Request Handler (Pydantic Models)    │  │
│  └────────────────┬──────────────────────┘  │
│                   │                          │
│  ┌────────────────▼──────────────────────┐  │
│  │  ThreadPoolExecutor (Async Wrapper)   │  │
│  └────────────────┬──────────────────────┘  │
│                   │                          │
│  ┌────────────────▼──────────────────────┐  │
│  │  DeepSeek-OCR Model (3B params)       │  │
│  │  - Flash Attention 2                  │  │
│  │  - BF16 Precision                     │  │
│  └────────────────┬──────────────────────┘  │
└───────────────────┼──────────────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   NVIDIA GPU    │
          │  (CUDA 11.8+)   │
          └─────────────────┘
```

### Key Design Decisions

1. **ThreadPoolExecutor for Async**: The DeepSeek-OCR model's `.infer()` method is synchronous. To prevent blocking the FastAPI event loop, we use `ThreadPoolExecutor` with `asyncio.run_in_executor()` to run inference in a thread pool.

2. **Temporary File Management**: Images received as base64 are decoded and saved to temporary files (required by the model), then cleaned up using FastAPI's `BackgroundTasks`.

3. **BF16 Precision**: Uses Brain Float 16 (BF16) precision to reduce memory usage while maintaining model quality.

4. **Flash Attention 2**: Leverages Flash Attention 2 for faster inference and reduced memory consumption.

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space (for model and dependencies)

### Software
- **OS**: Ubuntu 22.04 LTS (or compatible Linux distribution)
- **Python**: 3.12+
- **CUDA**: 11.8 or higher
- **NVIDIA Driver**: Compatible with CUDA 11.8+

## Installation

### Quick Start

1. **Clone the repository** (or transfer files to your server):
```bash
git clone <repository-url>
cd deepseek-ocr
```

2. **Run the setup script**:
```bash
chmod +x setup.sh run_server.sh
./setup.sh
```

The setup script will:
- Detect your environment (Conda or venv)
- Validate Python version (requires 3.12+)
- Install PyTorch 2.6.0 with CUDA 11.8 support
- Install transformers, tokenizers, and other dependencies
- Compile Flash Attention 2 from source (takes 10-20 minutes)
- Create necessary directories

3. **Wait for installation to complete**:
   - Total installation time: 15-30 minutes
   - Flash Attention compilation: 10-20 minutes
   - First model download: 5-10 minutes (happens on first server start)

### Manual Installation

If you prefer manual installation:

```bash
# Create conda environment
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr

# Install PyTorch with CUDA
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers
pip install transformers==4.46.3 tokenizers==0.20.3

# Install Flash Attention (this takes time)
pip install flash-attn==2.7.3 --no-build-isolation

# Install FastAPI and other dependencies
pip install fastapi uvicorn[standard] pydantic pillow python-multipart aiofiles

# Create directories
mkdir -p logs outputs
```

## Usage

### Starting the Server

```bash
# Activate environment (if using conda)
conda activate deepseek-ocr

# Start the server
./run_server.sh
```

The server will:
1. Activate the conda/venv environment
2. Validate dependencies
3. Check CUDA availability
4. Download the model (first time only, ~6GB)
5. Load the model into GPU memory
6. Start listening on `http://0.0.0.0:8000`

**Note**: First startup takes 2-5 minutes while the model downloads and loads.

### Server Configuration

Configure the server using environment variables:

```bash
# Custom configuration
export MODEL_NAME="deepseek-ai/DeepSeek-OCR"
export HOST="0.0.0.0"
export PORT="8000"
export BASE_SIZE="1024"
export IMAGE_SIZE="640"
export CROP_MODE="True"
export MAX_WORKERS="4"

./run_server.sh
```

### Stopping the Server

Press `Ctrl+C` to gracefully shutdown the server.

## API Endpoints

### 1. Health Check

Check server status and GPU availability.

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090"
}
```

### 2. General OCR

Extract text from an image.

```bash
POST /ocr
```

**Request Body:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "prompt": "<image>\nFree OCR.",
  "base_size": 1024,
  "image_size": 640,
  "crop_mode": true
}
```

**Response:**
```json
{
  "success": true,
  "result": "Extracted text from the image...",
  "metadata": {
    "prompt": "<image>\nFree OCR.",
    "output_path": null
  }
}
```

### 3. Document to Markdown

Convert a document image to markdown format.

```bash
POST /document-to-markdown
```

**Request Body:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "base_size": 1024,
  "image_size": 640,
  "crop_mode": true
}
```

### 4. Parse Figure

Parse diagrams, charts, and figures.

```bash
POST /parse-figure
```

**Request Body:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "base_size": 1024,
  "image_size": 640
}
```

### 5. Locate Text

Find specific text within an image.

```bash
POST /locate-text
```

**Request Body:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "text_to_locate": "Introduction",
  "base_size": 1024,
  "image_size": 640
}
```

### 6. Upload OCR

Upload an image file for OCR processing.

```bash
POST /upload-ocr
```

**Form Data:**
- `file`: Image file (PNG, JPG, etc.)
- `task`: Task type (`ocr`, `markdown`, `figure`)

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/upload-ocr" \
  -F "file=@document.png" \
  -F "task=markdown"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `deepseek-ai/DeepSeek-OCR` | Hugging Face model name |
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `8000` | Server port |
| `BASE_SIZE` | `1024` | Base resolution for image processing |
| `IMAGE_SIZE` | `640` | Target image size |
| `CROP_MODE` | `True` | Enable image cropping |
| `MAX_WORKERS` | `4` | Thread pool size for async inference |

### Model Resolution Settings

The model supports multiple resolution modes:

- **512×512**: 64 tokens (fastest, lowest quality)
- **640×640**: 100 tokens (default, balanced)
- **1024×1024**: 256 tokens (high quality)
- **1280×1280**: 400 tokens (highest quality, slowest)

## Testing

### Using the Test Client

The included `test_client.py` provides a convenient way to test all endpoints:

```bash
# Test general OCR
python test_client.py --image sample.png --task ocr

# Test document to markdown
python test_client.py --image document.png --task markdown

# Test figure parsing
python test_client.py --image diagram.png --task figure

# Test text localization
python test_client.py --image page.png --task locate --text "Chapter 1"

# Run all tests
python test_client.py --image test.png --task all

# Test remote server
python test_client.py --server http://192.168.10.3:8000 --image test.png --task ocr
```

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# OCR with file upload
curl -X POST "http://localhost:8000/upload-ocr" \
  -F "file=@test.png" \
  -F "task=ocr"

# Document to markdown with base64
python -c "import base64; print(base64.b64encode(open('doc.png', 'rb').read()).decode())" > image_b64.txt
curl -X POST "http://localhost:8000/document-to-markdown" \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$(cat image_b64.txt)\"}"
```

### Using Python Requests

```python
import requests
import base64

# Encode image
with open('test.png', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    'http://localhost:8000/ocr',
    json={'image_base64': image_base64}
)

print(response.json())
```

## Performance

### Model Loading

- **First run**: 2-5 minutes (downloads ~6GB model from Hugging Face)
- **Subsequent runs**: 30-60 seconds (loads from cache)
- **GPU VRAM usage**: 8-10GB (BF16 precision)

### Inference Speed

Performance varies by hardware and image complexity:

| Task | Estimated Time | Throughput |
|------|----------------|------------|
| Simple OCR | 1-3 seconds | ~20-60 req/min |
| Document to Markdown | 2-5 seconds | ~12-30 req/min |
| Figure Parsing | 2-4 seconds | ~15-30 req/min |

**Note**: These are estimates. Actual performance depends on:
- GPU model and VRAM
- Image size and complexity
- Concurrent requests
- Network latency

### Optimization Tips

1. **Batch Processing**: Use `MAX_WORKERS` to parallelize requests
2. **Image Size**: Smaller images process faster (use appropriate `image_size`)
3. **Resolution**: Lower resolution modes are faster but less accurate
4. **GPU**: Better GPU = faster inference (A100 > RTX 3090 > RTX 3060)

## Deployment

### Remote Server Deployment

1. **Transfer files to server**:
```bash
scp -r . ksnyder@192.168.10.3:/home/ksnyder/deepseek-ocr
```

2. **SSH into server**:
```bash
ssh ksnyder@192.168.10.3
cd /home/ksnyder/deepseek-ocr
```

3. **Run setup**:
```bash
chmod +x setup.sh run_server.sh
./setup.sh
```

4. **Start server**:
```bash
./run_server.sh
```

5. **Test from local machine**:
```bash
python test_client.py --server http://192.168.10.3:8000 --image test.png --task all
```

### Production Considerations

1. **Process Management**: Use systemd or supervisor to keep server running
2. **Reverse Proxy**: Put nginx or Caddy in front for HTTPS
3. **Monitoring**: Set up logging and metrics collection
4. **Scaling**: Use load balancer for multiple GPU servers
5. **Security**: Add authentication and rate limiting

### Systemd Service (Optional)

Create `/etc/systemd/system/deepseek-ocr.service`:

```ini
[Unit]
Description=DeepSeek OCR Server
After=network.target

[Service]
Type=simple
User=ksnyder
WorkingDirectory=/home/ksnyder/deepseek-ocr
ExecStart=/home/ksnyder/deepseek-ocr/run_server.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable deepseek-ocr
sudo systemctl start deepseek-ocr
sudo systemctl status deepseek-ocr
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `BASE_SIZE` and `IMAGE_SIZE`
- Lower `MAX_WORKERS` to reduce concurrent processing
- Use a GPU with more VRAM
- Process smaller images

#### 2. Flash Attention Installation Fails

**Symptom**: `flash-attn` compilation errors

**Solutions**:
- Ensure CUDA and NVIDIA drivers are properly installed
- Check GCC version (needs 7.0+)
- Try without flash-attn (model will use standard attention)
- Install build dependencies: `sudo apt install build-essential`

#### 3. Model Download Slow/Fails

**Symptom**: Model download times out or is very slow

**Solutions**:
- Check internet connection
- Set Hugging Face cache: `export HF_HOME=/path/to/cache`
- Download manually and set `HF_HOME`
- Use Hugging Face mirror if available

#### 4. Server Won't Start

**Symptom**: Server fails to start

**Solutions**:
- Check Python version: `python3 --version` (needs 3.12+)
- Verify environment activation
- Check port availability: `lsof -i :8000`
- Review logs in `logs/` directory

#### 5. Slow Inference

**Symptom**: Inference takes longer than expected

**Solutions**:
- Verify GPU is being used: check logs for "CUDA available: True"
- Monitor GPU usage: `nvidia-smi`
- Reduce image resolution
- Check for CPU bottlenecks

## What I Learned

Building this project taught me several valuable lessons about deploying AI models in production:

### 1. **Model Integration is Non-Trivial**

Unlike standard transformers models that use `.generate()`, DeepSeek-OCR has a custom `.infer()` method. This required:
- Reading model documentation carefully
- Understanding the model's specific API
- Using `trust_remote_code=True` for custom model code
- Handling model-specific parameters (base_size, image_size, crop_mode)

**Lesson**: Always read the model documentation first. Don't assume all models follow the same patterns.

### 2. **Async/Await in Python Requires Care**

FastAPI is async, but the model inference is synchronous. Key learnings:
- Can't directly `await` synchronous functions
- `ThreadPoolExecutor` is the right tool for CPU/GPU-bound sync operations
- `asyncio.run_in_executor()` bridges sync and async worlds
- Proper thread pool sizing prevents resource exhaustion

**Lesson**: Understand the difference between I/O-bound and CPU/GPU-bound operations. Choose the right async pattern for each.

### 3. **Resource Management is Critical**

Temporary files and GPU memory require careful management:
- Base64 images must be decoded to files (model requirement)
- Files must be cleaned up to prevent disk exhaustion
- FastAPI's `BackgroundTasks` is perfect for cleanup
- GPU memory is limited and must be monitored

**Lesson**: Always plan for resource cleanup. Temporary resources become permanent problems without proper management.

### 4. **Flash Attention is Powerful but Finicky**

Flash Attention 2 provides significant performance benefits but:
- Requires compilation from source (10-20 minutes)
- Can fail on systems without proper build tools
- Needs specific CUDA versions
- Model should gracefully fall back if unavailable

**Lesson**: Provide fallback options for optional optimizations. Don't make them hard requirements.

### 5. **Developer Experience Matters**

Creating good tooling saves time and frustration:
- Setup scripts reduce onboarding friction
- Colored output improves script readability
- Clear error messages with solutions help debugging
- Test clients make API testing easier
- Documentation prevents repetitive questions

**Lesson**: Invest in developer experience early. Good tooling pays dividends throughout the project lifecycle.

### 6. **API Design Impacts Usability**

Task-specific endpoints provide better UX than generic ones:
- `/document-to-markdown` is clearer than `/ocr` with a prompt parameter
- Sensible defaults reduce cognitive load
- Supporting both base64 and file upload increases flexibility
- Health checks enable monitoring and debugging

**Lesson**: Design APIs for the user, not just the implementation. Convenience matters.

### 7. **Configuration Should Be Flexible**

Environment variables provide the right balance:
- Easy to change without code modification
- Works well with containers and deployment tools
- Allows per-environment configuration
- Documented defaults prevent confusion

**Lesson**: Make configuration explicit and discoverable. Good defaults with easy overrides.

### 8. **Documentation is Code**

Good documentation:
- Reduces support burden
- Enables self-service
- Captures decisions and rationale
- Helps future maintainers (including future you)

**Lesson**: Write documentation while building, not after. Document the "why" not just the "what".

### 9. **Testing Early Prevents Problems**

Building the test client alongside the server:
- Validates API design decisions early
- Provides usage examples
- Makes debugging easier
- Serves as integration tests

**Lesson**: Build testing tools as you build features. They'll catch problems before users do.

### 10. **Performance is a Feature**

Optimization matters for production deployments:
- BF16 precision reduces memory usage without quality loss
- Proper async handling prevents blocking
- Thread pool sizing impacts throughput
- GPU utilization determines cost-effectiveness

**Lesson**: Design for performance from the start. Retrofitting optimization is harder than building it in.

## Future Enhancements

Potential improvements for future versions:

1. **Batch Processing**: Add endpoint for processing multiple images in one request
2. **Streaming Responses**: Stream results for large documents
3. **PDF Support**: Direct PDF processing without image conversion
4. **Result Caching**: Cache results for identical images
5. **Rate Limiting**: Protect server from abuse
6. **Authentication**: Add API key or JWT authentication
7. **Webhooks**: Async processing with callback URLs
8. **Metrics**: Prometheus metrics for monitoring
9. **Docker**: Containerize for easier deployment
10. **vLLM Integration**: Explore vLLM for higher throughput

## License

This project is provided as-is for educational and commercial use. The DeepSeek-OCR model is licensed under the MIT License by DeepSeek AI.

## Acknowledgments

- **DeepSeek AI** for the DeepSeek-OCR model
- **Hugging Face** for model hosting and transformers library
- **FastAPI** team for the excellent web framework
- **PyTorch** team for the deep learning framework

---

**Questions or Issues?** Please open an issue on GitHub or contact the maintainer.

**Want to Contribute?** Pull requests are welcome! Please read the contribution guidelines first.
