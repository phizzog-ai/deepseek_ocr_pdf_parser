#!/usr/bin/env python3
"""
DeepSeek OCR FastAPI Server

A FastAPI server for serving the DeepSeek-OCR model for optical character recognition tasks.
Supports multiple OCR operations including document-to-markdown conversion, figure parsing,
and text localization.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from transformers import AutoModel, AutoTokenizer
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
import io
import os
import sys
import tempfile
import logging
from pathlib import Path
from contextlib import redirect_stdout
import uvicorn
from PIL import Image
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
BASE_SIZE = int(os.getenv("BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
CROP_MODE = os.getenv("CROP_MODE", "True").lower() == "true"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek OCR API Server",
    description="OCR service using DeepSeek-OCR model for document understanding and text extraction",
    version="1.0.0"
)

# Global model and tokenizer
model = None
tokenizer = None
executor = None

# Pydantic models for request/response validation
class OCRRequest(BaseModel):
    """Request model for OCR operations"""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image (not implemented)")
    prompt: Optional[str] = Field(None, description="Custom prompt for OCR task")
    base_size: Optional[int] = Field(BASE_SIZE, description="Base resolution for processing")
    image_size: Optional[int] = Field(IMAGE_SIZE, description="Target image size")
    crop_mode: Optional[bool] = Field(CROP_MODE, description="Enable image cropping")
    save_results: bool = Field(False, description="Save results to disk")
    test_compress: bool = Field(True, description="Test compression")

class DocumentToMarkdownRequest(BaseModel):
    """Request model for document-to-markdown conversion"""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    base_size: Optional[int] = Field(BASE_SIZE, description="Base resolution for processing")
    image_size: Optional[int] = Field(IMAGE_SIZE, description="Target image size")
    crop_mode: Optional[bool] = Field(CROP_MODE, description="Enable image cropping")

class ParseFigureRequest(BaseModel):
    """Request model for figure parsing"""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    base_size: Optional[int] = Field(BASE_SIZE, description="Base resolution for processing")
    image_size: Optional[int] = Field(IMAGE_SIZE, description="Target image size")

class LocateTextRequest(BaseModel):
    """Request model for text localization"""
    image_base64: str = Field(..., description="Base64 encoded image")
    text_to_locate: str = Field(..., description="Text to locate in the image")
    base_size: Optional[int] = Field(BASE_SIZE, description="Base resolution for processing")
    image_size: Optional[int] = Field(IMAGE_SIZE, description="Target image size")

class OCRResponse(BaseModel):
    """Response model for OCR operations"""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize model and tokenizer on server startup"""
    global model, tokenizer, executor

    logger.info(f"Loading DeepSeek-OCR model: {MODEL_NAME}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Load model
        logger.info("Loading model (this may take a few minutes)...")
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        )

        # Move to GPU and set to eval mode
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model moved to GPU")

        model = model.eval()
        model = model.to(torch.bfloat16)
        logger.info("Model loaded successfully in BF16 precision")

        # Initialize thread pool for async inference
        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        logger.info(f"Thread pool executor initialized with {MAX_WORKERS} workers")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    global executor
    if executor:
        executor.shutdown(wait=True)
        logger.info("Thread pool executor shut down")

def decode_base64_image(base64_string: str) -> str:
    """
    Decode base64 image and save to temporary file

    Args:
        base64_string: Base64 encoded image

    Returns:
        Path to temporary file
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.write(image_data)
        temp_file.close()

        return temp_file.name
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

def run_inference(
    prompt: str,
    image_path: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    save_results: bool,
    test_compress: bool
) -> Dict[str, Any]:
    """
    Run inference on the model (synchronous function to be called in thread pool)

    Args:
        prompt: Prompt for the OCR task
        image_path: Path to image file
        base_size: Base resolution for processing
        image_size: Target image size
        crop_mode: Enable cropping
        save_results: Save results to disk
        test_compress: Test compression

    Returns:
        Dictionary containing inference results
    """
    try:
        # Create output directory
        output_path = tempfile.mkdtemp()

        logger.info(f"Running inference with prompt: {prompt}")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Parameters: base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")

        # Capture stdout since model.infer() prints OCR text to stdout
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            # Run model inference
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_path,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=save_results,
                test_compress=test_compress
            )

        # Get the captured output (this is the actual OCR text)
        ocr_text = captured_output.getvalue()

        logger.info("Inference completed successfully")
        logger.info(f"Captured OCR text length: {len(ocr_text)} characters")

        return {
            "success": True,
            "result": ocr_text if ocr_text else result,  # Use captured text, fallback to result
            "output_path": output_path if save_results else None
        }
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

async def async_inference(
    prompt: str,
    image_path: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    save_results: bool,
    test_compress: bool
) -> Dict[str, Any]:
    """
    Async wrapper for model inference

    Args:
        prompt: Prompt for the OCR task
        image_path: Path to image file
        base_size: Base resolution for processing
        image_size: Target image size
        crop_mode: Enable cropping
        save_results: Save results to disk
        test_compress: Test compression

    Returns:
        Dictionary containing inference results
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        run_inference,
        prompt,
        image_path,
        base_size,
        image_size,
        crop_mode,
        save_results,
        test_compress
    )
    return result

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "DeepSeek OCR API",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "status": "ready" if model is not None else "initializing",
        "endpoints": {
            "/health": "Health check",
            "/ocr": "General OCR endpoint",
            "/document-to-markdown": "Convert document to markdown",
            "/parse-figure": "Parse figures and diagrams",
            "/locate-text": "Locate text in image",
            "/upload-ocr": "Upload file for OCR"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.post("/ocr", response_model=OCRResponse)
async def ocr(request: OCRRequest, background_tasks: BackgroundTasks):
    """
    General OCR endpoint with custom prompt support

    Args:
        request: OCR request parameters
        background_tasks: FastAPI background tasks for cleanup

    Returns:
        OCR results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    # Default prompt for free OCR
    prompt = request.prompt or "<image>\nFree OCR."

    # Decode image
    image_path = decode_base64_image(request.image_base64)
    background_tasks.add_task(cleanup_temp_file, image_path)

    # Run inference
    result = await async_inference(
        prompt=prompt,
        image_path=image_path,
        base_size=request.base_size,
        image_size=request.image_size,
        crop_mode=request.crop_mode,
        save_results=request.save_results,
        test_compress=request.test_compress
    )

    if result["success"]:
        return OCRResponse(
            success=True,
            result=result["result"],
            metadata={
                "output_path": result.get("output_path"),
                "prompt": prompt
            }
        )
    else:
        return OCRResponse(
            success=False,
            error=result["error"]
        )

@app.post("/document-to-markdown", response_model=OCRResponse)
async def document_to_markdown(request: DocumentToMarkdownRequest, background_tasks: BackgroundTasks):
    """
    Convert document image to markdown format

    Args:
        request: Document conversion request
        background_tasks: FastAPI background tasks for cleanup

    Returns:
        Markdown conversion results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    prompt = "<image>\n<|grounding|>Convert the document to markdown."

    # Decode image
    image_path = decode_base64_image(request.image_base64)
    background_tasks.add_task(cleanup_temp_file, image_path)

    # Run inference
    result = await async_inference(
        prompt=prompt,
        image_path=image_path,
        base_size=request.base_size,
        image_size=request.image_size,
        crop_mode=request.crop_mode,
        save_results=False,
        test_compress=True
    )

    if result["success"]:
        return OCRResponse(
            success=True,
            result=result["result"],
            metadata={"prompt": prompt, "task": "document-to-markdown"}
        )
    else:
        return OCRResponse(
            success=False,
            error=result["error"]
        )

@app.post("/parse-figure", response_model=OCRResponse)
async def parse_figure(request: ParseFigureRequest, background_tasks: BackgroundTasks):
    """
    Parse figures and diagrams

    Args:
        request: Figure parsing request
        background_tasks: FastAPI background tasks for cleanup

    Returns:
        Figure parsing results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    prompt = "<image>\nParse the figure."

    # Decode image
    image_path = decode_base64_image(request.image_base64)
    background_tasks.add_task(cleanup_temp_file, image_path)

    # Run inference
    result = await async_inference(
        prompt=prompt,
        image_path=image_path,
        base_size=request.base_size,
        image_size=request.image_size,
        crop_mode=False,
        save_results=False,
        test_compress=True
    )

    if result["success"]:
        return OCRResponse(
            success=True,
            result=result["result"],
            metadata={"prompt": prompt, "task": "parse-figure"}
        )
    else:
        return OCRResponse(
            success=False,
            error=result["error"]
        )

@app.post("/locate-text", response_model=OCRResponse)
async def locate_text(request: LocateTextRequest, background_tasks: BackgroundTasks):
    """
    Locate specific text in the image

    Args:
        request: Text localization request
        background_tasks: FastAPI background tasks for cleanup

    Returns:
        Text localization results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = f"<image>\nLocate <|ref|>{request.text_to_locate}<|/ref|> in the image."

    # Decode image
    image_path = decode_base64_image(request.image_base64)
    background_tasks.add_task(cleanup_temp_file, image_path)

    # Run inference
    result = await async_inference(
        prompt=prompt,
        image_path=image_path,
        base_size=request.base_size,
        image_size=request.image_size,
        crop_mode=False,
        save_results=False,
        test_compress=False
    )

    if result["success"]:
        return OCRResponse(
            success=True,
            result=result["result"],
            metadata={"prompt": prompt, "task": "locate-text", "search_text": request.text_to_locate}
        )
    else:
        return OCRResponse(
            success=False,
            error=result["error"]
        )

@app.post("/upload-ocr")
async def upload_ocr(
    file: UploadFile = File(...),
    task: str = "ocr",
    background_tasks: BackgroundTasks = None
):
    """
    Upload a file for OCR processing

    Args:
        file: Uploaded image file
        task: OCR task type (ocr, markdown, figure)
        background_tasks: FastAPI background tasks for cleanup

    Returns:
        OCR results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    content = await file.read()
    temp_file.write(content)
    temp_file.close()

    background_tasks.add_task(cleanup_temp_file, temp_file.name)

    # Select prompt based on task
    prompts = {
        "ocr": "<image>\nFree OCR.",
        "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "figure": "<image>\nParse the figure."
    }
    prompt = prompts.get(task, prompts["ocr"])

    # Run inference
    result = await async_inference(
        prompt=prompt,
        image_path=temp_file.name,
        base_size=BASE_SIZE,
        image_size=IMAGE_SIZE,
        crop_mode=CROP_MODE,
        save_results=False,
        test_compress=True
    )

    if result["success"]:
        return OCRResponse(
            success=True,
            result=result["result"],
            metadata={"prompt": prompt, "task": task, "filename": file.filename}
        )
    else:
        return OCRResponse(
            success=False,
            error=result["error"]
        )

if __name__ == "__main__":
    logger.info(f"Starting DeepSeek OCR server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
