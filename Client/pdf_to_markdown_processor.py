#!/usr/bin/env python3
"""
PDF to Markdown Processor (Enhanced)

This application scans the /data folder for PDF files and converts them to Markdown format
using the DeepSeek OCR API at localhost:8000. Each PDF file is converted to a Markdown 
file with the same name in the same /data folder.

Enhanced version includes post-processing steps from run_dpsk_ocr_pdf.py:
- Special token cleanup
- Reference processing for layout information
- Image extraction and markdown link generation
- Content cleaning and formatting
"""

import os
import sys
import glob
import logging
import base64
import json
import requests
import re
import io
import tempfile
import urllib.parse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image, ImageDraw
import numpy as np
import fitz  # PyMuPDF

# Try to import PDF processing libraries
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not available. Install with: pip install pdf2image")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 not available. Install with: pip install PyPDF2")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


class PDFToMarkdownProcessor:
    """Processor for converting PDF files to Markdown using DeepSeek OCR API with enhanced post-processing"""
    
    def __init__(self, data_folder: str = "data", api_base_url: str = "http://192.168.10.3:8000",
                 extract_images: bool = True, create_images_folder: bool = True):
        """
        Initialize the PDF processor
        
        Args:
            data_folder: Path to the folder containing PDF files
            api_base_url: Base URL of the DeepSeek OCR API
            extract_images: Whether to extract images from the PDF
            create_images_folder: Whether to create an images subfolder for extracted images
        """
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        self.api_base_url = api_base_url
        self.extract_images = extract_images
        self.create_images_folder = create_images_folder
        
        # Create images subfolder if needed
        if self.extract_images and self.create_images_folder:
            self.images_folder = self.data_folder / "images"
            self.images_folder.mkdir(exist_ok=True)
        else:
            self.images_folder = None
        
        # Test API connection
        if not self._test_api_connection():
            raise ConnectionError(f"Cannot connect to API at {api_base_url}")
    
    def _test_api_connection(self) -> bool:
        """Test if the API is accessible"""
        try:
            response = requests.get(f"{self.api_base_url}/docs", timeout=5)
            if response.status_code == 200:
                logger.info("API connection successful")
                return True
            else:
                logger.error(f"API returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"API connection failed: {str(e)}")
            return False
    
    def _get_api_endpoints(self) -> Dict[str, str]:
        """Get available API endpoints"""
        try:
            response = requests.get(f"{self.api_base_url}/openapi.json", timeout=5)
            if response.status_code == 200:
                openapi_spec = response.json()
                endpoints = {}
                for path, methods in openapi_spec.get("paths", {}).items():
                    for method, details in methods.items():
                        if method.upper() in ["POST", "GET"]:
                            operation_id = details.get("operationId", "")
                            if "pdf" in operation_id.lower() or "ocr" in operation_id.lower():
                                endpoints[operation_id] = f"{method.upper()} {path}"
                return endpoints
            else:
                logger.error(f"Failed to get API spec: {response.status_code}")
                return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting API spec: {str(e)}")
            return {}
    
    def _pdf_to_images(self, pdf_path: str, dpi: int = 144) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of PIL Images
        """
        images = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                
                # Convert to PIL Image
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            
            pdf_document.close()
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
        
        return images
    
    def _re_match(self, text: str) -> Tuple[List, List, List]:
        """
        Match reference patterns in the text
        
        Args:
            text: The text to search for patterns
            
        Returns:
            Tuple of (all_matches, image_matches, other_matches)
        """
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        matches_image = []
        matches_other = []
        
        for a_match in matches:
            if '<|ref|>image<|/ref|>' in a_match[0]:
                matches_image.append(a_match[0])
            else:
                matches_other.append(a_match[0])
        
        return matches, matches_image, matches_other
    
    def _extract_coordinates_and_label(self, ref_text: Tuple) -> Optional[Tuple[str, List]]:
        """
        Extract coordinates and label from reference text
        
        Args:
            ref_text: Reference text tuple from regex match
            
        Returns:
            Tuple of (label_type, coordinates_list) or None if extraction fails
        """
        try:
            label_type = ref_text[1]
            cor_list = eval(ref_text[2])
            return (label_type, cor_list)
        except Exception as e:
            logger.error(f"Error extracting coordinates: {str(e)}")
            return None
    
    def _ocr_image(self, image_path: str) -> Optional[str]:
        """
        OCR an extracted image using the /parse-figure endpoint

        Args:
            image_path: Path to the image file

        Returns:
            OCR'd text from the image, or None if failed
        """
        try:
            # Convert image to base64
            with open(image_path, 'rb') as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            # Call /parse-figure endpoint
            endpoint = "/parse-figure"
            url = f"{self.api_base_url}{endpoint}"

            payload = {
                "image_base64": image_base64,
                "base_size": 1024,
                "image_size": 640,
                "crop_mode": True
            }

            response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                if result.get("success") and result.get("result"):
                    return result["result"]
            else:
                logger.warning(f"Failed to OCR image {image_path}: {response.status_code}")

            return None

        except Exception as e:
            logger.error(f"Error OCR'ing image {image_path}: {str(e)}")
            return None

    def _extract_and_save_images(self, pdf_path: str, content: str, page_idx: int) -> Tuple[str, int]:
        """
        Extract images from content, OCR them, and save to the images folder

        Args:
            pdf_path: Path to the original PDF file
            content: The OCR content with reference tags
            page_idx: Index of the page being processed

        Returns:
            Tuple of (processed_content, number_of_images_extracted)
        """
        if not self.extract_images or not self.images_folder:
            return content, 0

        # Get PDF images for this page
        pdf_images = self._pdf_to_images(pdf_path)
        if page_idx >= len(pdf_images):
            return content, 0

        page_image = pdf_images[page_idx]
        image_width, image_height = page_image.size

        # Find all image references
        _, matches_images, _ = self._re_match(content)
        img_idx = 0

        for idx, a_match_image in enumerate(matches_images):
            try:
                # Extract the reference text
                pattern = r'<\|ref\|>image<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
                det_match = re.search(pattern, a_match_image)

                if det_match:
                    det_content = det_match.group(1)
                    try:
                        coordinates = eval(det_content)

                        # Extract and save the image
                        for points in coordinates:
                            x1, y1, x2, y2 = points

                            # Scale coordinates to actual image size
                            x1 = int(x1 / 999 * image_width)
                            y1 = int(y1 / 999 * image_height)
                            x2 = int(x2 / 999 * image_width)
                            y2 = int(y2 / 999 * image_height)

                            # Crop and save the image
                            cropped = page_image.crop((x1, y1, x2, y2))
                            image_filename = f"{Path(pdf_path).stem}_page{page_idx}_{img_idx}.jpg"
                            image_path = self.images_folder / image_filename
                            cropped.save(image_path)

                            logger.info(f"OCR'ing extracted image: {image_filename}")

                            # OCR the extracted image to get its content
                            image_ocr_text = self._ocr_image(str(image_path))

                            # Build the replacement content
                            encoded_filename = urllib.parse.quote(image_filename)

                            if image_ocr_text:
                                # Include both the OCR'd text AND the image reference
                                replacement = f"\n\n**[Figure/Chart Content]**\n\n{image_ocr_text}\n\n![](images/{encoded_filename})\n\n"
                            else:
                                # If OCR failed, just include the image
                                replacement = f"\n\n![](images/{encoded_filename})\n\n"

                            content = content.replace(a_match_image, replacement, 1)

                            img_idx += 1
                            break
                    except Exception as e:
                        logger.error(f"Error processing image coordinates: {str(e)}")
                        # If we can't process the coordinates, just remove the tag
                        content = content.replace(a_match_image, "", 1)
            except Exception as e:
                logger.error(f"Error extracting image: {str(e)}")
                content = content.replace(a_match_image, "", 1)

        return content, img_idx
    
    def _pretty_print_html_table(self, html_table: str) -> str:
        """
        Pretty-print HTML table for better readability

        Args:
            html_table: Single-line HTML table string

        Returns:
            Formatted HTML table with proper indentation
        """
        try:
            # Add newlines and indentation to table tags
            formatted = html_table.replace('<table>', '<table>\n  ')
            formatted = formatted.replace('</table>', '\n</table>')
            formatted = formatted.replace('<tr>', '<tr>\n    ')
            formatted = formatted.replace('</tr>', '\n  </tr>')
            formatted = formatted.replace('<td>', '<td>')
            formatted = formatted.replace('</td>', '</td>')
            formatted = formatted.replace('</tr><tr>', '</tr>\n  <tr>')
            formatted = formatted.replace('</td><td>', '</td><td>')

            return formatted
        except Exception as e:
            logger.warning(f"Error pretty-printing table: {e}")
            return html_table

    def _clean_content(self, content: str) -> str:
        """
        Clean up the OCR content

        Args:
            content: Raw OCR content

        Returns:
            Cleaned content
        """
        # Remove end of sentence tokens
        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')

        # Remove debug output patterns
        # Pattern 1: ===== BASE: torch.Size(...) =====
        content = re.sub(r'={5,}\n.*?torch\.Size.*?\n={5,}\n?', '', content, flags=re.DOTALL)

        # Pattern 2: BASE: torch.Size(...) and PATCHES: torch.Size(...)
        content = re.sub(r'BASE:\s+torch\.Size\([^\)]+\)\n', '', content)
        content = re.sub(r'PATCHES:\s+torch\.Size\([^\)]+\)\n', '', content)
        content = re.sub(r'NO PATCHES\n', '', content)

        # Pattern 3: directly resize
        content = re.sub(r'directly resize\n', '', content)

        # Pattern 4: Image size and compression stats
        content = re.sub(r'={40,}\n.*?image size:.*?\n.*?valid image tokens:.*?\n.*?compression ratio:.*?\n={40,}\n?', '', content, flags=re.DOTALL)

        # Get all non-image references
        _, _, matches_other = self._re_match(content)

        # Remove other reference tags and clean up
        for idx, a_match_other in enumerate(matches_other):
            content = content.replace(a_match_other, '')

        # Pretty-print HTML tables
        table_pattern = r'<table>.*?</table>'
        tables = re.findall(table_pattern, content, re.DOTALL)
        for table in tables:
            if '<tr>' in table:  # Only process if it looks like a real table
                pretty_table = self._pretty_print_html_table(table)
                content = content.replace(table, pretty_table)

        # Replace special LaTeX-like symbols
        content = content.replace('\\coloneqq', ':=')
        content = content.replace('\\eqqcolon', '=:')

        # Fix LaTeX parentheses formatting
        content = re.sub(r'\\?\(([^)]+)\\?\)', r'(\1)', content)

        # Clean up excessive newlines
        content = content.replace('\n\n\n\n', '\n\n')
        content = content.replace('\n\n\n', '\n\n')

        return content.strip()
    
    def _process_page_content(self, pdf_path: str, content: str, page_idx: int) -> str:
        """
        Process a single page's content with all post-processing steps
        
        Args:
            pdf_path: Path to the original PDF file
            content: Raw OCR content for the page
            page_idx: Index of the page being processed
            
        Returns:
            Processed content
        """
        # Step 1: Extract and save images
        content, num_images = self._extract_and_save_images(pdf_path, content, page_idx)

        # Step 2: Clean up the content
        content = self._clean_content(content)

        # Step 3: Add page separator with page number
        page_number = page_idx + 1  # Convert 0-indexed to 1-indexed
        page_separator = f'\n\n<--- Page {page_number} End --->\n\n'
        content += page_separator

        logger.info(f"Processed page {page_idx + 1}, extracted {num_images} images")

        return content
    
    def _pdf_page_to_base64(self, page_number: int, pdf_path: str, dpi: int = 200) -> Optional[str]:
        """
        Convert PDF page to base64 encoded image using pdf2image

        Args:
            page_number: Page number (1-indexed)
            pdf_path: Path to PDF file
            dpi: DPI for rendering

        Returns:
            Base64 encoded image string
        """
        if not PDF2IMAGE_AVAILABLE:
            logger.error("pdf2image is required for page conversion")
            return None

        try:
            # Convert single page
            images = convert_from_path(
                pdf_path,
                first_page=page_number,
                last_page=page_number,
                dpi=dpi
            )

            if not images:
                return None

            # Convert to base64
            img_buffer = io.BytesIO()
            images[0].save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            return img_base64

        except Exception as e:
            logger.error(f"Error converting page {page_number}: {e}")
            return None

    def _get_pdf_page_count(self, pdf_path: str) -> int:
        """
        Get total page count from PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages
        """
        if PYPDF2_AVAILABLE:
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    return len(pdf_reader.pages)
            except Exception as e:
                logger.warning(f"Could not get page count with PyPDF2: {e}")

        # Fallback to PyMuPDF
        try:
            pdf_document = fitz.open(pdf_path)
            page_count = pdf_document.page_count
            pdf_document.close()
            return page_count
        except Exception as e:
            logger.error(f"Error getting page count: {e}")
            return 0

    def _call_ocr_api(self, pdf_path: str) -> Optional[str]:
        """
        Call the OCR API to process a PDF file page by page

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Markdown content or None if processing failed
        """
        try:
            # Get total page count
            total_pages = self._get_pdf_page_count(pdf_path)

            if total_pages == 0:
                logger.error(f"Could not determine page count for {pdf_path}")
                return None

            logger.info(f"Processing PDF: {pdf_path} ({total_pages} pages)")

            # Use the /document-to-markdown endpoint for page-by-page processing
            endpoint = "/document-to-markdown"
            url = f"{self.api_base_url}{endpoint}"

            processed_content = ""

            # Process each page
            for page_num in range(1, total_pages + 1):
                logger.info(f"Processing page {page_num}/{total_pages}...")

                # Convert page to base64
                image_base64 = self._pdf_page_to_base64(page_num, pdf_path)

                if not image_base64:
                    logger.error(f"Failed to convert page {page_num} to image")
                    continue

                # Call OCR API for this page
                payload = {
                    "image_base64": image_base64,
                    "base_size": 1024,
                    "image_size": 640,
                    "crop_mode": True
                }

                response = requests.post(url, json=payload, timeout=120)

                if response.status_code == 200:
                    result = response.json()

                    if result.get("success") and result.get("result"):
                        page_content = result["result"]

                        # Apply post-processing to this page
                        processed_page = self._process_page_content(
                            pdf_path, page_content, page_num - 1  # 0-indexed for processing
                        )
                        processed_content += processed_page
                    else:
                        logger.warning(f"Page {page_num} returned no content")
                else:
                    logger.error(f"API request failed for page {page_num}: {response.status_code}")

            if not processed_content:
                logger.error("No content was successfully processed")
                return None

            logger.info(f"Successfully processed all {total_pages} pages")
            return processed_content.strip()

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return None
    
    def convert_pdf_to_markdown(self, pdf_path: str) -> Optional[str]:
        """
        Convert a single PDF file to Markdown
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the generated Markdown file, or None if conversion failed
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Call OCR API
            markdown_content = self._call_ocr_api(pdf_path)
            
            if not markdown_content:
                logger.error(f"Failed to get markdown content for {pdf_path}")
                return None
            
            # Save markdown file with -MD suffix
            pdf_path_obj = Path(pdf_path)
            markdown_path = pdf_path_obj.with_name(f"{pdf_path_obj.stem}-MD.md")
            
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Successfully converted {pdf_path} to {markdown_path}")
            return str(markdown_path)
            
        except Exception as e:
            logger.error(f"Error converting {pdf_path}: {str(e)}")
            return None
    
    def scan_and_process_all_pdfs(self) -> List[str]:
        """
        Scan the data folder for PDF files and convert all of them to Markdown
        
        Returns:
            List of paths to generated Markdown files
        """
        # Find all PDF files in the data folder
        pdf_files = list(self.data_folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.info(f"No PDF files found in {self.data_folder}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        markdown_files = []
        for pdf_file in pdf_files:
            markdown_file = self.convert_pdf_to_markdown(str(pdf_file))
            if markdown_file:
                markdown_files.append(markdown_file)
        
        return markdown_files


def main():
    """Main function to run the PDF processor"""
    print(f"{Colors.BLUE}PDF to Markdown Processor (Enhanced){Colors.RESET}")
    print(f"{Colors.YELLOW}Scanning /data folder for PDF files...{Colors.RESET}")
    
    try:
        processor = PDFToMarkdownProcessor(
            extract_images=True,
            create_images_folder=True
        )
        markdown_files = processor.scan_and_process_all_pdfs()
        
        if markdown_files:
            print(f"\n{Colors.GREEN}Successfully converted {len(markdown_files)} PDF files to Markdown:{Colors.RESET}")
            for md_file in markdown_files:
                print(f"  - {md_file}")
            if processor.extract_images:
                print(f"\n{Colors.BLUE}Images extracted to: {processor.images_folder}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}No PDF files were processed.{Colors.RESET}")
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()