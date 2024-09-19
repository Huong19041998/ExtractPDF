# PDF Image Extractor

## Description

This program helps you analyze and extract images from PDF files. It handles both scanned PDFs and raw PDFs using different tools based on the type of PDF.

## Features

- **Determine PDF Type**: Identifies whether the PDF is a scanned document based on metadata, content, and OCR.
- **Extract Images**: Extracts images from PDFs using OpenCV for scanned PDFs and PyMuPDF for raw PDFs.
- **Save Results**: Saves extracted images to a specified output directory.

## Installation

**Install Dependencies**:

   ```bash
   pip install -r requirements.txt
```
## Usage
Run the program using the following command:
   ```bash
   python pdf_image_extractor.py example.pdf
```
Where example.pdf is the path to the PDF file you want to process.

#### Parameters
- pdf_path: Path to the PDF file to process.  
- output_dir: Directory to save the extracted images. (Default is images)  
- tracing_enabled: Enable or disable tracing mode. (Default is True)  
- tracing_dir: Directory to save tracing images if tracing is enabled. (Default is tracing)  

#### Example
To process a PDF and save images to the output_images directory with tracing enabled:
   ```bash
   python pdf_image_extractor.py example.pdf --output_dir output_images --tracing_enabled True --tracing_dir trace_images
   ```
