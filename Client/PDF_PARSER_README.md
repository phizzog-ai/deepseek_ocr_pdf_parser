# PDF Parser - Complete Pipeline

A complete pipeline for converting PDFs into structured markdown and JSON with OCR, image extraction, and table parsing.

## 🎯 Overview

This pipeline converts PDFs into two formats:
1. **Markdown** with extracted images and tables (human-readable)
2. **JSON** with structured data and base64-encoded images (machine-readable)

## 📊 Complete Flow

```
PDF Document
    ↓
[pdf_to_markdown_processor.py]
    ↓
Markdown + Images (data/images/)
    ↓
[markdown_to_json.py]
    ↓
JSON with base64 images
```

## 🚀 Quick Start

### 1. Process PDF to Markdown

```bash
# Put PDFs in the data folder
mkdir data
cp your_document.pdf data/

# Run the processor
python pdf_to_markdown_processor.py
```

**Output:**
```
data/
├── your_document.pdf
├── your_document-MD.md          ← Markdown output
└── images/
    ├── your_document_page0_0.jpg
    ├── your_document_page1_0.jpg
    └── ...
```

### 2. Convert Markdown to JSON

```bash
# Convert the markdown to JSON
python markdown_to_json.py data/your_document-MD.md
```

**Output:**
```
data/
└── your_document-MD.json        ← JSON output
```

## 📁 File Structure

```
deepseek-ocr/
├── pdf_to_markdown_processor.py    # PDF → Markdown + Images
├── markdown_to_json.py              # Markdown → JSON
├── data/
│   ├── example.pdf                  # Input PDF
│   ├── example-MD.md                # Markdown output
│   ├── example-MD.json              # JSON output
│   └── images/
│       ├── example_page0_0.jpg      # Extracted charts/figures
│       ├── example_page1_0.jpg
│       └── ...
├── README.md                        # Main project README
└── PDF_PARSER_README.md            # This file
```

## 🔧 Components

### Component 1: pdf_to_markdown_processor.py

**Purpose:** Convert PDFs to markdown with OCR and image extraction

**Features:**
- ✅ Page-by-page OCR using DeepSeek-OCR model
- ✅ Extract charts/tables as images using grounding coordinates
- ✅ OCR each extracted image to get data
- ✅ Generate clean markdown with embedded content
- ✅ Page number markers for navigation

**Usage:**
```python
from pdf_to_markdown_processor import PDFToMarkdownProcessor

processor = PDFToMarkdownProcessor(
    data_folder="data",
    api_base_url="http://192.168.10.3:8000",
    extract_images=True,
    create_images_folder=True
)

# Process all PDFs in folder
markdown_files = processor.scan_and_process_all_pdfs()

# Or process single PDF
markdown_file = processor.convert_pdf_to_markdown("data/example.pdf")
```

**Output Format:**
```markdown
## Chapter Title

Body text content...

**[Figure/Chart Content]**

<table>
  <tr>
    <td>Category</td><td>Value</td>
  </tr>
  <tr>
    <td>AI and information processing</td><td>86%</td>
  </tr>
</table>

![](images/example_page1_0.jpg)

<--- Page 1 End --->

More content...

<--- Page 2 End --->
```

**Performance:**
- ~8-10 seconds per page (includes OCR + image extraction)
- Processes sequentially (GPU-bound)

### Component 2: markdown_to_json.py

**Purpose:** Convert markdown to structured JSON format

**Features:**
- ✅ Parse pages by `<--- Page N End --->` markers
- ✅ Extract HTML tables into structured data
- ✅ Encode images as base64
- ✅ Clean text content
- ✅ Compatible with output_parsed.json schema

**Usage:**
```bash
# Basic usage
python markdown_to_json.py data/example-MD.md

# Specify output file
python markdown_to_json.py data/example-MD.md -o data/output.json

# Custom images folder
python markdown_to_json.py data/example-MD.md -i /path/to/images
```

**Programmatic Usage:**
```python
from markdown_to_json import MarkdownToJSONConverter

converter = MarkdownToJSONConverter(
    markdown_file="data/example-MD.md",
    images_folder="data/images"  # Optional
)

# Get JSON dict
result = converter.convert()

# Or save directly
converter.save_json("data/output.json")
```

**Output Format:**
```json
{
  "pdf_document": {
    "document_id": "doc_example",
    "filename": "example.pdf",
    "total_pages": 11,
    "metadata": {}
  },
  "pages": [
    {
      "page_id": "page_1",
      "pdf_title": "example.pdf",
      "text": "Clean text content without tables/images...",
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
        "/9j/4AAQSkZJRgABAQAAAQABAAD..."
      ]
    }
  ]
}
```

## 🎨 What Gets Extracted

### From PDFs:
- ✅ **Text content** (full OCR of all pages)
- ✅ **Tables** (converted to HTML/JSON format)
- ✅ **Charts & Figures** (extracted as images + OCR'd for data)
- ✅ **Grounding coordinates** (bounding boxes for elements)
- ✅ **Page structure** (headings, paragraphs, lists)

### Image Extraction Process:

1. **DeepSeek OCR detects chart location**
   ```
   <|ref|>image<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
   ```

2. **Crop from PDF page**
   ```python
   cropped = page_image.crop((x1, y1, x2, y2))
   cropped.save("images/example_page0_0.jpg")
   ```

3. **OCR the extracted image**
   ```
   Calls /parse-figure endpoint
   Returns table data: "AI: 86%, Robots: 58%..."
   ```

4. **Include both in output**
   ```markdown
   **[Figure/Chart Content]**
   <table>...</table>
   ![](images/example_page0_0.jpg)
   ```

## 📈 Example Results

### Input: example.pdf (11 pages)

**Processing Stats:**
- Processing time: ~90 seconds
- Pages processed: 11
- Tables extracted: 8
- Images extracted: 9
- Markdown output: 473 lines
- JSON output: 707KB

**Quality:**
- ✅ All text captured accurately
- ✅ All chart data extracted (86%, 58%, 41%, etc.)
- ✅ All tables formatted properly
- ✅ No debug output in final files
- ✅ Clean, professional formatting

## 🛠️ Dependencies

### Python Packages
```bash
pip install PyMuPDF pdf2image PyPDF2 Pillow requests
```

### System Requirements
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils
```

### Server Requirements
- DeepSeek OCR server running at http://192.168.10.3:8000
- See main README.md for server setup

## 🔍 Advanced Usage

### Custom Processing Options

```python
# Disable image extraction (faster)
processor = PDFToMarkdownProcessor(
    extract_images=False
)

# Process specific page range (requires modification)
# Currently processes all pages - could be extended

# Custom DPI for images
processor._pdf_page_to_base64(page_num, pdf_path, dpi=300)
```

### JSON Schema Customization

The output JSON matches `output_parsed.json` schema. To customize:

```python
# Modify the convert() method in MarkdownToJSONConverter
result = {
    "pdf_document": {
        "document_id": "custom_id",
        "filename": "custom.pdf",
        "total_pages": len(pages),
        "metadata": {
            "custom_field": "value"  # Add custom metadata
        }
    },
    "pages": pages
}
```

## 📊 Output Comparison

### Markdown Output (Human-Readable)
**Best for:**
- Reading and reviewing
- Manual editing
- Documentation
- Publishing to web

**Contains:**
- Formatted text with headings
- HTML tables (readable)
- Image file references
- Page markers

### JSON Output (Machine-Readable)
**Best for:**
- Data analysis
- API integration
- Database import
- Automated processing

**Contains:**
- Structured data objects
- Tables as arrays
- Base64-encoded images
- Metadata

## 🐛 Troubleshooting

### Issue: "No PDFs found in data folder"
**Solution:** Make sure PDFs are directly in the `data/` folder, not in subfolders

### Issue: "Cannot connect to API"
**Solution:** Verify DeepSeek OCR server is running:
```bash
curl http://192.168.10.3:8000/health
```

### Issue: "ModuleNotFoundError: No module named 'fitz'"
**Solution:** Reinstall PyMuPDF:
```bash
pip uninstall -y PyMuPDF fitz
pip install --no-cache-dir PyMuPDF
```

### Issue: "Images not found in JSON"
**Solution:** Ensure `images/` folder exists in same directory as markdown file

### Issue: "Empty tables in JSON"
**Solution:** Tables must have valid HTML structure with `<tr>` and `<td>` tags

## 🎯 Use Cases

### 1. Research Paper Processing
```bash
# Process academic papers
python pdf_to_markdown_processor.py  # Extracts all figures and tables
python markdown_to_json.py data/paper-MD.md  # For database import
```

### 2. Report Analysis
```bash
# Extract data from business reports
python pdf_to_markdown_processor.py
# Then analyze tables in JSON programmatically
```

### 3. Documentation Migration
```bash
# Convert old PDFs to markdown for web
python pdf_to_markdown_processor.py
# Edit markdown as needed
# Publish to docs site
```

### 4. Data Extraction
```bash
# Extract structured data from forms/invoices
python pdf_to_markdown_processor.py
python markdown_to_json.py data/invoice-MD.md
# Import JSON into database
```

## 🚦 Performance Tips

### Speed Optimization
1. **Disable image extraction** if you only need text:
   ```python
   processor = PDFToMarkdownProcessor(extract_images=False)
   ```

2. **Process in batches** for large collections

3. **Use lower DPI** for faster processing:
   ```python
   dpi=144  # Default is 200
   ```

### Quality Optimization
1. **Use higher DPI** for better OCR:
   ```python
   dpi=300  # Better for small text
   ```

2. **Keep images enabled** for complete data extraction

3. **Review markdown output** before converting to JSON

## 📝 Notes

- **Sequential Processing:** Pages are processed one at a time (GPU limitation)
- **Image Quality:** Extracted images are JPG format (balance of size/quality)
- **Table Detection:** Uses HTML table tags from OCR output
- **Page Markers:** Format is `<--- Page N End --->` for easy splitting
- **Base64 Encoding:** Images in JSON can be large (use compression if needed)

## 🔗 Related Files

- [README.md](README.md) - Main project documentation
- [PDF_PROCESSOR_UPDATE.md](PDF_PROCESSOR_UPDATE.md) - Detailed processor changes
- [.docs/notes.md](.docs/notes.md) - Implementation notes and lessons learned

## 📄 License

Same as main project (see main README.md)

## 🤝 Contributing

Improvements welcome! Areas for enhancement:
- Parallel page processing (when multiple GPUs available)
- Resume interrupted processing
- Progress bars for long documents
- Configurable output formats
- Table structure improvements
- Better error recovery

## 📞 Support

For issues or questions:
1. Check [.docs/notes.md](.docs/notes.md) for implementation details
2. Review server logs in `logs/` folder
3. Test with small PDFs first
4. Verify server connectivity

---

**Last Updated:** October 22, 2025
**Version:** 1.0.0
**Status:** Production Ready ✅
