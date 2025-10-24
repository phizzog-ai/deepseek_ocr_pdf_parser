# DeepSeek OCR PDF Parser

A production-ready system for converting PDF documents into structured markdown and JSON formats using the DeepSeek-OCR model. This project provides a GPU-accelerated pipeline for high-quality document digitization with table extraction, image extraction, and text localization capabilities.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This project consists of two main components:

1. **Server**: FastAPI-based OCR service powered by the DeepSeek-OCR model
2. **Client**: PDF processing pipeline that converts PDFs to markdown and JSON

The system transforms PDF documents through a multi-stage pipeline:
```
PDF → Images → OCR with Grounding → Markdown → Structured JSON
```

## Project Structure

```
deepseek_ocr_pdf_parser/
├── Server/                           # FastAPI OCR Service
│   ├── deepseek_ocr_server.py       # Main server (577 lines)
│   ├── setup.sh                      # Environment setup script
│   ├── run_server.sh                 # Server startup script
│   └── README.md                     # Detailed server documentation
│
├── Client/                           # PDF Processing Pipeline
│   ├── pdf_to_markdown_processor.py  # PDF to Markdown converter (671 lines)
│   ├── markdown_to_json.py           # Markdown to JSON converter (280 lines)
│   ├── data/                         # Input/output directory for PDFs
│   └── PDF_PARSER_README.md          # Client documentation
│
├── .docs/                            # Documentation
│   └── DeepSeek_OCR_paper.pdf        # Research paper reference
│
├── README.md                         # This file
├── LICENSE                           # Apache 2.0 License
└── .gitignore                        # Git ignore rules
```

## Architecture

### Data Flow

```
┌─────────────────┐
│   PDF Document  │
└────────┬────────┘
         │
         v
┌─────────────────────────────────────┐
│  PDFToMarkdownProcessor (Client)   │
│  1. Convert PDF pages to images     │
│  2. Base64 encode each page         │
└────────┬────────────────────────────┘
         │ POST /document-to-markdown
         v
┌─────────────────────────────────────┐
│  DeepSeek OCR Server (FastAPI)      │
│  1. Decode base64 to temp file      │
│  2. Run model inference (GPU)       │
│  3. Return OCR with grounding       │
└────────┬────────────────────────────┘
         │ Returns: Markdown + Coordinates
         v
┌─────────────────────────────────────┐
│  Image Extraction (Client)          │
│  1. Parse grounding coordinates     │
│  2. Crop images from original PDF   │
│  3. OCR extracted images            │
└────────┬────────────────────────────┘
         │
         v
┌─────────────────────────────────────┐
│  Markdown Output                    │
│  - Structured text                  │
│  - HTML tables                      │
│  - Image references                 │
│  - Page markers                     │
└────────┬────────────────────────────┘
         │
         v
┌─────────────────────────────────────┐
│  MarkdownToJSONConverter            │
│  1. Split by page markers           │
│  2. Extract tables and images       │
│  3. Clean text content              │
└────────┬────────────────────────────┘
         │
         v
┌─────────────────┐
│  JSON Output    │
│  - Structured   │
│  - Searchable   │
│  - Machine-     │
│    readable     │
└─────────────────┘
```

### Server Architecture

```
HTTP Request (Base64 Image)
         ↓
FastAPI Request Handler (Pydantic Validation)
         ↓
ThreadPoolExecutor (Async Wrapper)
         ↓
DeepSeek-OCR Model (GPU Inference)
         ↓
NVIDIA CUDA GPU (BF16 + Flash Attention 2)
         ↓
OCR Text with Grounding Coordinates
```

## Features

### Server Features

- **Multiple OCR Modes**:
  - Free OCR (extract all text)
  - Document to Markdown (with table detection)
  - Figure Parsing (charts, diagrams)
  - Text Localization (find specific text with coordinates)

- **Performance Optimizations**:
  - Brain Float 16 (BF16) precision for memory efficiency
  - Flash Attention 2 for faster inference
  - Async request handling with ThreadPoolExecutor
  - Automatic temporary file cleanup

- **Flexible Input**:
  - Base64 encoded images
  - File uploads (multipart form data)
  - Configurable image preprocessing

### Client Features

- **PDF Processing**:
  - Batch processing of multiple PDFs
  - Automatic image extraction using grounding coordinates
  - Page-by-page processing with progress tracking
  - Customizable DPI settings

- **Intelligent Extraction**:
  - Table detection and HTML formatting
  - Figure/chart extraction and OCR
  - Image cropping based on model coordinates
  - Content cleaning (removes debug tokens)

- **Output Formats**:
  - Markdown with embedded tables and images
  - Structured JSON with metadata
  - Base64-encoded images in JSON
  - Extracted images as separate JPEG files

## Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with 8-10GB VRAM (e.g., RTX 3090, A4000)
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ (model + dependencies)

### Software Requirements

- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.12+
- **CUDA**: 11.8+ (for GPU acceleration)

## Quick Start

### 1. Server Setup

```bash
cd Server

# Run automated setup (creates environment and installs dependencies)
./setup.sh

# Start the server
./run_server.sh
```

The server will:
- Download the DeepSeek-OCR model (~6GB) on first run
- Start on `http://0.0.0.0:8000`
- Log to timestamped files in `logs/`

### 2. Client Setup

```bash
cd Client

# Install dependencies
pip install PyMuPDF pdf2image PyPDF2 Pillow numpy requests

# Place your PDFs in the data/ folder
cp /path/to/your/document.pdf data/

# Process a single PDF
python pdf_to_markdown_processor.py

# Or process all PDFs in data/ folder
# (uncomment the batch processing line in the main() function)
```

### 3. Convert Markdown to JSON

```bash
cd Client

# Convert markdown to JSON
python markdown_to_json.py data/example-MD.md

# Specify output file
python markdown_to_json.py data/example-MD.md -o data/output.json

# Custom images folder
python markdown_to_json.py data/example-MD.md -i /path/to/images
```

## How It Works

### Step 1: PDF to Images

The client converts each PDF page to a high-resolution image using PyMuPDF:

```python
processor = PDFToMarkdownProcessor(
    data_folder="data",
    api_base_url="http://192.168.10.3:8000",
    extract_images=True
)
```

### Step 2: OCR with Grounding

Each page image is sent to the server for OCR. The model returns markdown text with special grounding coordinates:

```markdown
<|ref|>image<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
![](extracted_image.jpg)

<|ref|>table<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
<table>
  <tr><th>Category</th><th>Value</th></tr>
  <tr><td>AI</td><td>86%</td></tr>
</table>
```

### Step 3: Image Extraction

The client parses grounding coordinates and extracts images:

```python
# Coordinates are in 0-999 normalized range
x1_actual = int(x1 / 999 * page_width)
y1_actual = int(y1 / 999 * page_height)
x2_actual = int(x2 / 999 * page_width)
y2_actual = int(y2 / 999 * page_height)

# Crop and save
cropped_image = page_image.crop((x1_actual, y1_actual, x2_actual, y2_actual))
cropped_image.save(f"images/{filename}_page{idx}_{img_idx}.jpg")
```

### Step 4: Figure OCR

Extracted images are sent back to the server for detailed OCR:

```python
# Server parses the figure/chart
figure_text = call_api("/parse-figure", cropped_image_base64)
```

### Step 5: Content Cleaning

The client cleans the OCR output:
- Removes special tokens (`<｜end▁of▁sentence｜>`)
- Removes debug output (tensor sizes, compression stats)
- Pretty-prints HTML tables
- Fixes LaTeX symbols
- Adds page markers: `<--- Page N End --->`

### Step 6: Markdown to JSON

The converter structures the markdown into JSON:

```json
{
  "pdf_document": {
    "document_id": "doc_example",
    "filename": "example.pdf",
    "total_pages": 2,
    "metadata": {}
  },
  "pages": [
    {
      "page_id": "page_1",
      "pdf_title": "example.pdf",
      "text": "Cleaned page content...",
      "tables": [
        {
          "columns": ["Category", "Value"],
          "data": [["AI", "86%"]],
          "extends_to_bottom": false,
          "chart_data": true
        }
      ],
      "image_base64": ["/9j/4AAQSkZJRgABAQAA..."]
    }
  ]
}
```

## Usage

### Server Usage

#### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090"
}
```

#### Document to Markdown

```bash
curl -X POST http://localhost:8000/document-to-markdown \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": true
  }'
```

#### Parse Figure

```bash
curl -X POST http://localhost:8000/parse-figure \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "base_size": 1024,
    "image_size": 640
  }'
```

#### Locate Text

```bash
curl -X POST http://localhost:8000/locate-text \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "text_to_locate": "Machine Learning",
    "base_size": 1024,
    "image_size": 640
  }'
```

### Client Usage

#### Single PDF Processing

```python
from pdf_to_markdown_processor import PDFToMarkdownProcessor

# Initialize processor
processor = PDFToMarkdownProcessor(
    data_folder="data",
    api_base_url="http://192.168.10.3:8000",
    extract_images=True
)

# Convert PDF to markdown
markdown_file = processor.convert_pdf_to_markdown("data/example.pdf")
print(f"Generated: {markdown_file}")
```

#### Batch Processing

```python
# Process all PDFs in data/ folder
processor = PDFToMarkdownProcessor()
markdown_files = processor.scan_and_process_all_pdfs()

for md_file in markdown_files:
    print(f"Processed: {md_file}")
```

#### Markdown to JSON

```python
from markdown_to_json import MarkdownToJSONConverter
import json

# Convert markdown to JSON
converter = MarkdownToJSONConverter(
    markdown_file="data/example-MD.md",
    images_folder="data/images"
)

json_data = converter.convert()

# Save to file
with open("data/example-MD.json", "w") as f:
    json.dump(json_data, f, indent=2)
```

## API Reference

### Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and server status |
| `/ocr` | POST | General OCR with custom prompt |
| `/document-to-markdown` | POST | Convert document to markdown |
| `/parse-figure` | POST | Parse charts, diagrams, figures |
| `/locate-text` | POST | Find specific text with coordinates |
| `/upload-ocr` | POST | File upload for OCR |

### Request Models

#### OCRRequest
```python
{
  "image_base64": "string (optional)",
  "image_url": "string (optional)",
  "prompt": "string (optional)",
  "base_size": "int (default: 1024)",
  "image_size": "int (default: 640)",
  "crop_mode": "bool (default: true)",
  "save_results": "bool (default: false)",
  "test_compress": "bool (default: true)"
}
```

#### DocumentToMarkdownRequest
```python
{
  "image_base64": "string (optional)",
  "base_size": "int (default: 1024)",
  "image_size": "int (default: 640)",
  "crop_mode": "bool (default: true)"
}
```

#### ParseFigureRequest
```python
{
  "image_base64": "string (optional)",
  "base_size": "int (default: 1024)",
  "image_size": "int (default: 640)"
}
```

#### LocateTextRequest
```python
{
  "image_base64": "string (required)",
  "text_to_locate": "string (required)",
  "base_size": "int (default: 1024)",
  "image_size": "int (default: 640)"
}
```

### Response Model

```python
{
  "success": "bool",
  "result": "string (optional)",
  "error": "string (optional)",
  "metadata": "object (optional)"
}
```

## Configuration

### Server Environment Variables

Edit `Server/run_server.sh` to configure:

```bash
MODEL_NAME="deepseek-ai/DeepSeek-OCR"  # Hugging Face model
HOST="0.0.0.0"                         # Server host
PORT="8000"                            # Server port
BASE_SIZE="1024"                       # Base resolution
IMAGE_SIZE="640"                       # Target image size
CROP_MODE="True"                       # Enable cropping
MAX_WORKERS="4"                        # Thread pool size
```

### Client Configuration

Edit the client scripts to configure:

```python
# Server URL (currently hardcoded)
api_base_url = "http://192.168.10.3:8000"

# Image extraction
extract_images = True
create_images_folder = True

# Data directory
data_folder = "data"

# DPI for PDF conversion
dpi = 144  # Lower = faster, Higher = better quality
```

### Model Resolution Settings

| Resolution | Tokens | Quality | Speed | Use Case |
|-----------|--------|---------|-------|----------|
| 512×512 | 64 | Low | Fastest | Quick previews |
| 640×640 | 100 | Medium | Default | Balanced quality/speed |
| 1024×1024 | 256 | High | Slower | High-quality documents |
| 1280×1280 | 400 | Very High | Slowest | Research papers, detailed diagrams |

## Output Formats

### Markdown Output

Generated file: `data/example-MD.md`

```markdown
# Document Title

Regular text content with proper formatting.

## Tables

<table>
  <tr>
    <th>Category</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>AI and information processing</td>
    <td>86%</td>
  </tr>
</table>

## Images

![](images/example_page0_0.jpg)

**[Figure Content]**
This chart shows the distribution of AI technologies...

<--- Page 1 End --->

Second page content...

<--- Page 2 End --->
```

### Extracted Images

Saved to: `data/images/`

Filename format: `{pdf_name}_page{page_idx}_{img_idx}.jpg`

Example:
```
data/images/
├── example_page0_0.jpg
├── example_page0_1.jpg
├── example_page1_0.jpg
└── example_page2_0.jpg
```

### JSON Output

Generated file: `data/example-MD.json`

```json
{
  "pdf_document": {
    "document_id": "doc_example",
    "filename": "example.pdf",
    "total_pages": 3,
    "metadata": {}
  },
  "pages": [
    {
      "page_id": "page_1",
      "pdf_title": "example.pdf",
      "text": "Document Title\n\nRegular text content...",
      "tables": [
        {
          "columns": ["Category", "Value"],
          "data": [
            ["AI and information processing", "86%"],
            ["Robots and autonomous systems", "58%"]
          ],
          "extends_to_bottom": false,
          "chart_data": true
        }
      ],
      "image_base64": [
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAg..."
      ]
    }
  ]
}
```

## Troubleshooting

### Server Issues

#### Model Not Loading
```
ERROR: Model failed to load
```

**Solutions**:
- Check CUDA installation: `nvidia-smi`
- Verify GPU memory: Need 8-10GB free
- Check Python version: Must be 3.12+
- Review logs in `Server/logs/`

#### First Inference Very Slow
```
First inference takes 2-5 minutes
```

**This is normal**: The model performs warmup compilation on first run. Subsequent inferences will be 1-3 seconds.

#### Out of Memory
```
CUDA out of memory error
```

**Solutions**:
- Reduce `IMAGE_SIZE` in config (try 512)
- Reduce `BASE_SIZE` to 768
- Close other GPU applications
- Reduce `MAX_WORKERS` to 2

### Client Issues

#### Connection Refused
```
ConnectionError: Could not connect to OCR server
```

**Solutions**:
- Check server is running: `curl http://192.168.10.3:8000/health`
- Update `api_base_url` to correct IP/port
- Check firewall settings

#### Images Not Extracted
```
No images found in markdown output
```

**Solutions**:
- Ensure `extract_images=True` in processor
- Check `images_folder` exists
- Verify OCR returned grounding coordinates
- Review logs for coordinate parsing errors

#### Poor OCR Quality
```
Text is garbled or missing
```

**Solutions**:
- Increase DPI in PDF conversion (try 200)
- Increase `IMAGE_SIZE` to 1024
- Increase `BASE_SIZE` to 1280
- Check input PDF quality

### JSON Conversion Issues

#### Missing Tables
```
Tables not appearing in JSON
```

**Solutions**:
- Check markdown contains `<table>` tags
- Verify HTML table format is valid
- Review `TableHTMLParser` logs

#### Missing Images
```
image_base64 arrays are empty
```

**Solutions**:
- Ensure images were extracted to `images/` folder
- Check `images_folder` path is correct
- Verify image files are readable

## Performance Tips

### Server Optimization

1. **Batch Similar Resolutions**: Process documents of similar sizes together
2. **Adjust Workers**: Increase `MAX_WORKERS` for more concurrent requests (limited by GPU memory)
3. **Use Lower Resolution**: For draft processing, use 512x512
4. **Enable Flash Attention**: Significantly faster with `flash-attn` installed

### Client Optimization

1. **Lower DPI for Speed**: Use 144 DPI instead of 200
2. **Skip Image Extraction**: Set `extract_images=False` if not needed
3. **Process in Batches**: Process multiple PDFs overnight
4. **Use SSD Storage**: Faster image reading/writing

### Typical Performance

| Document Type | Pages | Processing Time | Output Size |
|---------------|-------|-----------------|-------------|
| Simple Text | 10 | 2-3 min | 50KB markdown |
| Mixed Content | 10 | 5-8 min | 200KB markdown + images |
| Image-Heavy | 10 | 10-15 min | 1MB markdown + images |
| Research Paper | 20 | 15-25 min | 500KB markdown + images |

## Additional Resources

- [Server README](Server/README.md) - Detailed server documentation (705 lines)
- [Client README](Client/PDF_PARSER_README.md) - Client pipeline documentation (473 lines)
- [DeepSeek OCR Paper](.docs/DeepSeek_OCR_paper.pdf) - Research paper reference

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **DeepSeek-AI** for the DeepSeek-OCR model
- **FastAPI** for the excellent web framework
- **Hugging Face** for model hosting and transformers library

## Support

For issues or questions:
1. Check the detailed component READMEs
2. Review the troubleshooting section above
3. Check server logs in `Server/logs/`
4. Verify your configuration matches the requirements

---

**Note**: This is a research/development tool. For production use, consider adding:
- Authentication and rate limiting
- HTTPS support (reverse proxy)
- Request validation and timeouts
- Monitoring and metrics
- Docker containerization
- Load balancing for multiple GPUs
