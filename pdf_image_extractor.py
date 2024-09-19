import fitz  # PyMuPDF
from pathlib import Path
import os
import cv2
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import argparse

class PDFImageExtractor:
    def __init__(self, pdf_path, output_dir='images', tracing_enabled=True, tracing_dir='tracing'):
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
        scanners = ['TOSHIBA e-STUDIO', 'Canon', 'HP ScanJet', 'Epson']
        office_tools = ['Microsoft Office Word', 'Adobe Acrobat', 'LibreOffice']
        converters = ['PDFsharp', 'Ghostscript', 'PDFCreator']

        creator = metadata.get('creator', '')
        producer = metadata.get('producer', '')

        if any(scanner in creator for scanner in scanners) or any(scanner in producer for scanner in scanners):
            return True
        if any(tool in creator for tool in office_tools) or any(tool in producer for tool in office_tools):
            return False
        if any(converter in creator for converter in converters) or any(converter in producer for converter in converters):
            return False

        return None  # Unknown

    def is_pdf_scan_based_on_content(self):
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
        images = convert_from_path(self.pdf_path, first_page=1, last_page=1)
        for image in images:
            text = pytesseract.image_to_string(image)
            if text.strip():
                return False  # OCR found text, likely not a scan
        return True  # OCR did not find text, likely a scan

    def is_pdf_scan(self):
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

                image_filename = os.path.join(self.output_dir, f'page_{page_num + 1}_image_{img_index + 1}.{image_ext}')
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)
                print(f"Extracted {image_filename}")
        doc.close()

    def extract_images_from_opencv(self, input_filename):
        """Extract images from a page image using OpenCV."""
        minimum_width = 100
        minimum_height = 100
        green_color = (36, 255, 12)
        trace_width = 2

        image = cv2.imread(input_filename)
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        ROI_number = 1
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w >= minimum_width and h >= minimum_height:
                cv2.rectangle(image, (x, y), (x + w, y + h), green_color, trace_width)
                ROI = original[y:y + h, x:x + w]
                out_image = os.path.join(self.output_dir, f'{Path(input_filename).stem}_{ROI_number}.png')
                cv2.imwrite(out_image, ROI)
                ROI_number += 1

        if self.tracing_enabled:
            trace_image = os.path.join(self.tracing_dir, Path(input_filename).stem + '_trace.png')
            cv2.imwrite(trace_image, image)

    def convert_pdf_to_images(self):
        """Convert each page of the PDF to an image."""
        images = convert_from_path(self.pdf_path)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(self.output_dir, f"page_{i + 1}.png")
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
        return image_paths

    def process_pdf(self):
        """Process the PDF based on its type (scan or raw)."""
        if self.is_pdf_scan():
            print("Detected a scanned PDF. Processing with OpenCV...")
            image_files = self.convert_pdf_to_images()
            for image_file in image_files:
                self.extract_images_from_opencv(image_file)
        else:
            print("Detected a raw PDF. Processing with PyMuPDF...")
            self.extract_images_from_pymupdf()

def main():
    parser = argparse.ArgumentParser(description="Extract images from PDF files.")
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file.')
    parser.add_argument('--output_dir', type=str, default='images', help='Directory to save extracted images.')
    parser.add_argument('--tracing_enabled', action='store_true', help='Enable tracing of image extraction.')
    parser.add_argument('--tracing_dir', type=str, default='tracing', help='Directory to save tracing images.')

    args = parser.parse_args()

    extractor = PDFImageExtractor(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        tracing_enabled=args.tracing_enabled,
        tracing_dir=args.tracing_dir
    )
    extractor.process_pdf()

if __name__ == "__main__":
    main()
