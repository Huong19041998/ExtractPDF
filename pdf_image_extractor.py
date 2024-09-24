import fitz  # PyMuPDF
from pathlib import Path
import os
import cv2
import pytesseract
from PyPDF2 import PdfReader
import argparse
import numpy as np

class PDFImageExtractor:
    def __init__(self, pdf_path, output_dir='images', tracing_enabled=True, tracing_dir='tracing'):
        # Initialize the PDFImageExtractor with the provided parameters
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.tracing_enabled = tracing_enabled
        self.tracing_dir = tracing_dir
        self.prepare_directories()

    def prepare_directories(self):
        """Create output and tracing directories if they don't exist."""
        Path(self.output_dir).mkdir(exist_ok=True)
        if self.tracing_enabled:
            Path(self.tracing_dir).mkdir(exist_ok=True)

    def get_pdf_metadata(self):
        """Retrieve metadata from the PDF file."""
        with open(self.pdf_path, "rb") as f:
            pdf = PdfReader(f)
            info = pdf.metadata
            return {
                'creator': info.get('/Creator', ''),
                'producer': info.get('/Producer', ''),
                'creation_date': info.get('/CreationDate', ''),
                'mod_date': info.get('/ModDate', '')
            }

    def is_pdf_scan_based_on_metadata(self, metadata):
        """Determine if the PDF is scanned based on its metadata."""
        scanners = ['TOSHIBA e-STUDIO', 'Canon', 'HP ScanJet', 'Epson']
        office_tools = ['Microsoft Office Word', 'Adobe Acrobat', 'LibreOffice']
        converters = ['PDFsharp', 'Ghostscript', 'PDFCreator']

        creator = metadata.get('creator', '')
        producer = metadata.get('producer', '')

        if any(scanner in creator for scanner in scanners) or any(scanner in producer for scanner in scanners):
            return True  # PDF is likely scanned
        if any(tool in creator for tool in office_tools) or any(tool in producer for tool in office_tools):
            return False  # PDF is likely not scanned
        if any(converter in creator for converter in converters) or any(converter in producer for converter in converters):
            return False  # PDF is likely not scanned

        return None  # Unknown

    def is_pdf_scan_based_on_content(self):
        """Determine if the PDF is scanned based on its content."""
        doc = fitz.open(self.pdf_path)

        for page_num in range(min(3, len(doc))):
            page = doc.load_page(page_num)
            images = page.get_images(full=True)
            text = page.get_text()

            if not text.strip() and images:
                return True  # Likely a scan
            elif images and len(text) < 100:
                return True  # Likely a scan
            elif not images and len(text) > 0:
                return False  # Likely not a scan

        return None  # Unknown

    def is_pdf_scan_with_ocr(self):
        """Determine if the PDF is scanned by performing OCR on its images."""
        images = self.convert_pdf_to_images()
        for image in images:
            text = pytesseract.image_to_string(image)
            if text.strip():
                return False  # OCR found text, likely not a scan
        return True  # OCR did not find text, likely a scan

    def is_pdf_scan(self):
        """Determine if the PDF is scanned using various methods."""
        metadata = self.get_pdf_metadata()
        # Check based on metadata
        result = self.is_pdf_scan_based_on_metadata(metadata)
        if result is not None:
            return result  # Return True if metadata indicates scan, else False

        # Check based on content
        result = self.is_pdf_scan_based_on_content()
        if result is not None:
            return result  # Return True if content analysis indicates scan, else False

        # Check with OCR
        return self.is_pdf_scan_with_ocr()  # Return True if OCR indicates scan, else False

    def extract_images_from_pymupdf(self):
        """Extract images from a PDF using PyMuPDF."""
        doc = fitz.open(self.pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Construct the filename for the extracted image
                image_filename = os.path.join(self.output_dir, f'page_{page_num + 1}_image_{img_index + 1}.{image_ext}')
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)  # Write the image bytes to a file
                print(f"Extracted {image_filename}")
        doc.close()

    def extract_images_from_opencv(self, input_filename):
        """Extract regions of interest (ROIs) from a page image using OpenCV."""
        minimum_width = 100
        minimum_height = 100
        green_color = (36, 255, 12)
        trace_width = 2

        image = cv2.imread(input_filename)  # Read the input image
        original = image.copy()  # Keep a copy of the original image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Binarize the image

        ROI_number = 1
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # Find contours
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)  # Get bounding box of each contour
            if w >= minimum_width and h >= minimum_height:
                cv2.rectangle(image, (x, y), (x + w, y + h), green_color, trace_width)  # Draw rectangle around ROI
                ROI = original[y:y + h, x:x + w]  # Extract ROI from the original image
                out_image = os.path.join(self.output_dir, f'{Path(input_filename).stem}_{ROI_number}.png')
                cv2.imwrite(out_image, ROI)  # Save the ROI image
                ROI_number += 1

        # Save tracing image if tracing is enabled
        if self.tracing_enabled:
            trace_image = os.path.join(self.tracing_dir, Path(input_filename).stem + '_trace.png')
            cv2.imwrite(trace_image, image)  # Save the traced image

    def convert_pdf_to_images(self):
        """Convert each page of the PDF to an image using PyMuPDF."""
        doc = fitz.open(self.pdf_path)
        image_paths = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=300)  # Get a pixmap of the page
            image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
            image_path = os.path.join(self.output_dir, f"page_{i + 1}.png")
            cv2.imwrite(image_path, image_array)  # Save the image
            image_paths.append(image_path)
        return image_paths

    def process_pdf(self):
        """Process the PDF based on its type (scan or raw)."""
        if self.is_pdf_scan():
            print("Detected a scanned PDF. Processing with OpenCV...")
            image_files = self.convert_pdf_to_images()  # Convert PDF pages to images
            for image_file in image_files:
                self.extract_images_from_opencv(image_file)  # Extract ROIs from images
        else:
            print("Detected a raw PDF. Processing with PyMuPDF...")
            self.extract_images_from_pymupdf()  # Extract images directly from the PDF

def main():
    # Set up argument parser for command-line interface
    parser = argparse.ArgumentParser(description="Extract images from PDF files.")
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file.')  # PDF file path argument
    parser.add_argument('--output_dir', type=str, default='images', help='Directory to save extracted images.')  # Output directory
    parser.add_argument('--tracing_enabled', action='store_true', help='Enable tracing of image extraction.')  # Tracing flag
    parser.add_argument('--tracing_dir', type=str, default='tracing', help='Directory to save tracing images.')  # Tracing directory

    args = parser.parse_args()  # Parse arguments

    # Create an instance of PDFImageExtractor and process the PDF
    extractor = PDFImageExtractor(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        tracing_enabled=args.tracing_enabled,
        tracing_dir=args.tracing_dir
    )
    extractor.process_pdf()  # Process the PDF

if __name__ == "__main__":
    main()  # Run the main function
