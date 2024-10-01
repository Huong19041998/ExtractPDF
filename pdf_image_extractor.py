import fitz  # PyMuPDF
from pathlib import Path
import os
import cv2
import pytesseract
from PyPDF2 import PdfReader
import argparse
from paddleocr import PaddleOCR
import numpy as np

class PDFImageExtractor:
    def __init__(self, pdf_path, output_dir='images', tracing_enabled=True, tracing_dir='tracing', use_pymupdf_for_caption=True):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.tracing_enabled = tracing_enabled
        self.tracing_dir = tracing_dir
        self.prepare_directories()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.caption_ratio = 0.25  #Ratio for determining the caption area below images
        self.images_dict = {}  # Dictionary to store extracted images
        self.captions_dict = {}  # Dictionary to store captions for each image
        self.use_pymupdf_for_caption = use_pymupdf_for_caption

    def prepare_directories(self):
        """Create output and tracing directories if they do not exist."""
        # Create the output directory to store extracted images, if it doesn't already exist
        Path(self.output_dir).mkdir(exist_ok=True)
        # If tracing is enabled, create the tracing directory to store trace images
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
        if any(converter in creator for converter in converters) or any(
                converter in producer for converter in converters):
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
        """Extract images from the PDF using PyMuPDF."""
        doc = fitz.open(self.pdf_path)  # Open the PDF file with PyMuPDF
        for page_num in range(len(doc)):  # Iterate over each page in the PDF
            page = doc.load_page(page_num)  # Load the current page
            images = page.get_images(full=True)  # Get all images on the page
            pix = page.get_pixmap(dpi=72)  # Render the page as an image with a DPI of 72
            img_page = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width,
                                                                          pix.n)  # Convert the pixmap to a numpy array
            img_page = cv2.cvtColor(img_page, cv2.COLOR_RGB2BGR)  # Convert the image from RGB to BGR color space
            if self.use_pymupdf_for_caption:
                captions_page,caption_boxes = self.extract_captions_with_pymupdf(page)
            for img_index, img in enumerate(images):  # Iterate over each image found on the page
                rect = page.get_image_rects(img)[0]  # Get the rectangle coordinates of the image
                x0, y0, x1, y1 = int(rect.x0), int(rect.y0), int(rect.x1), int(
                    rect.y1)  # Convert coordinates to integers

                # Crop the image based on the coordinates
                cropped_image = img_page[int(y0):int(y1), int(x0):int(x1)]

                # Calculate the height for the caption area
                caption_height = int((y1 - y0) * self.caption_ratio)
                y2_caption = y1 + caption_height
                y2_caption = min(y2_caption,
                                 img_page.shape[0])  # Ensure the caption height doesn't exceed the image height

                # Adjust caption boundaries to include some padding
                x0_caption = 0 if 0 <= x0 < 10 else x0 - 10
                x1_caption = x1 if img_page.shape[1] - 10 <= x1 <= img_page.shape[1] else min(x1 + 10,
                                                                                              img_page.shape[1])

                # Crop the caption area from the image
                cropped_cap_image = img_page[y1:y2_caption, x0_caption:x1_caption]

                # Save the cropped image and caption image
                image_filename = os.path.join(self.output_dir, f'page_{page_num + 1}_cropped_image_{img_index + 1}.png')
                cv2.imwrite(image_filename, cropped_image)
                image_filename_cap = os.path.join(self.output_dir,
                                                  f'page_{page_num + 1}_cropped_image_cap_{img_index + 1}.png')
                cv2.imwrite(image_filename_cap, cropped_cap_image)

                # Extract the caption from the cropped caption image
                if self.use_pymupdf_for_caption:
                    caption = []
                    for index_cap ,caption_box in enumerate(caption_boxes):
                        if self.is_text_caption_for_image((x0, y0, x1, y1), caption_box):
                            caption.append(captions_page[index_cap])

                else:
                    caption = self.extract_caption_from_image(cropped_cap_image)



                self.images_dict[f"page_{page_num + 1}_{img_index + 1}"] = cropped_cap_image  # Store the cropped image
                self.captions_dict[f"page_{page_num + 1}_{img_index + 1}"] = caption  # Store the caption

        doc.close()  # Close the PDF document

    def extract_images_from_opencv(self, index_page, image_page):
        """Extract images from the PDF page using OpenCV."""
        minimum_width = 100  # Minimum width for the bounding rectangle of detected images
        minimum_height = 100  # Minimum height for the bounding rectangle of detected images
        # image = cv2.imread(image_page)
        # original = image_page.copy()
        gray = cv2.cvtColor(image_page, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Apply thresholding to create a binary image

        cropped_cap_image_number = 1  # Counter for cropped caption images
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # Find contours in the thresholded image
        for c in cnts:  # Iterate over each contour found
            x0, y0, w, h = cv2.boundingRect(c)  # Get the bounding rectangle for the contour
            x1, y1 = x0 + w, y0 + h  # Calculate the bottom-right coordinates of the bounding rectangle
            cropped_image = image_page[y0:y1, x0:x1]
            # Calculate the height for the caption area
            caption_height = int(h * self.caption_ratio)
            y2_caption = y0 + h + caption_height
            y2_caption = min(y2_caption, image_page.shape[0])  # Ensure the caption height doesn't exceed the image height

            # Adjust caption boundaries to include some padding
            x0_caption = 0 if 0 <= x0 < 10 else x0 - 10
            x1_caption = x1 if image_page.shape[1] - 10 <= x1 <= image_page.shape[1] else min(x1 + 10, image_page.shape[1])

            # Check if the width and height of the bounding rectangle meet the minimum requirements
            if w >= minimum_width and h >= minimum_height:
                # Crop the caption image based on the calculated coordinates
                cropped_cap_image = image_page[y1:y2_caption, x0_caption:x1_caption]
                # out_image_cap = os.path.join(self.output_dir,
                #                              f'page_{index_page + 1}_cap_{cropped_cap_image_number}.png')
                # cv2.imwrite(out_image_cap, cropped_cap_image)  # Save the cropped caption image
                out_image = os.path.join(self.output_dir, f'page_{index_page + 1}_{cropped_cap_image_number}.png')
                cv2.imwrite(out_image, cropped_image)  # Save the original cropped image

                # Extract the caption from the cropped caption image
                caption = self.extract_caption_from_image(cropped_cap_image)
                self.images_dict[
                    f"page_{index_page + 1}_{cropped_cap_image_number}"] = cropped_image  # Store the cropped caption image
                self.captions_dict[f"page_{index_page + 1}_{cropped_cap_image_number}"] = caption  # Store the caption
                cropped_cap_image_number += 1  # Increment the cropped caption image counter

        # If tracing is enabled, save a trace image of the original input image
        if self.tracing_enabled:
            trace_image = os.path.join(self.tracing_dir, f"page_{index_page + 1}_trace.png")
            cv2.imwrite(trace_image, image_page)  # Save the trace image
    def extract_captions_with_pymupdf(self, page):
        """Extract captions based on the image positions using PyMuPDF."""
        captions_page = []
        box_captions = []
        text_instances = page.get_text("dict")

        # Loop through the text instances to find captions
        for block in text_instances['blocks']:
            if block['type'] == 0:  # Type 0 means it's a text block
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text']
                        bbox = span['bbox']
                        captions_page.append(text)
                        box_captions.append(bbox)

        return captions_page,box_captions

    def is_text_caption_for_image(self,image_box, text_box):
        """
        Check if a text_box is a caption for an image_box
        based on position criteria.

        image_box: tuple (x0_img, y0_img, x1_img, y1_img) - Bounding box of the image.
        text_box: tuple (x0_text, y0_text, x1_text, y1_text) - Bounding box of the text.
        threshold: percentage of the image height that the caption's top y-coordinate should not exceed.

        Returns:
            True if the text_box is a caption for the image_box, False otherwise.
        """
        threshold = self.caption_ratio
        # Unpack the coordinates of the image_box and text_box
        x0_img, y0_img, x1_img, y1_img = image_box
        x0_text, y0_text, x1_text, y1_text = text_box

        # Condition 1: Check if the text_box is below the image_box
        if y1_img <= y0_text <= y1_img +  int((y1_img - y0_img) * self.caption_ratio) and x0_img - 100 <=  x0_text <= + 100 :
            # Calculate the height of the image_box
            image_height = y1_img - y0_img
            # Condition 2:  Check if the height of the text_box does not exceed 25% of the image_box height
            if (y1_text - y0_text) <= threshold * image_height:
                return True

        return False

    def extract_caption_from_image(self, image):
        """Use PaddleOCR to recognize captions from the image."""
        result = self.ocr.ocr(image, cls=True)
        captions = []
        if result:
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        captions.append(line[1][0])
        # Return the concatenated captions as a single string or None if no captions were found
        return ' '.join(captions) if captions else None

    def convert_pdf_to_images(self):
        """Convert each page of the PDF to an image using PyMuPDF."""
        doc = fitz.open(self.pdf_path)
        image_pages = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=72)  # Get a pixmap of the page
            image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
            image_path = os.path.join(self.output_dir, f"page_{i + 1}.png")
            cv2.imwrite(image_path, image_array)  # Save the image
            image_pages.append(image_array)
        return image_pages

    def process_pdf(self):
        """Process the PDF based on its type (scan or raw)."""
        if self.is_pdf_scan():
            print("Detected a scanned PDF. Processing with OpenCV...")
            image_pages = self.convert_pdf_to_images()
            for index_page, image_page in enumerate(image_pages):
                self.extract_images_from_opencv(index_page,image_page)
        else:
            print("Detected a raw PDF. Processing with PyMuPDF...")
            self.extract_images_from_pymupdf()
        return self.images_dict, self.captions_dict


def main():
    parser = argparse.ArgumentParser(description="Trích xuất hình ảnh và chú thích từ PDF.")
    parser.add_argument('pdf_path', type=str, help='Đường dẫn tới file PDF.')
    parser.add_argument('--output_dir', type=str, default='images', help='Thư mục để lưu hình ảnh đã trích xuất.')
    parser.add_argument('--tracing_enabled', action='store_true', help='Bật chế độ tracing.')
    parser.add_argument('--tracing_dir', type=str, default='tracing', help='Thư mục để lưu ảnh tracing.')
    parser.add_argument('--use_pymupdf_for_caption', type=bool, default=True,
                        help='If True, use PyMuPDF to extract captions for images.')

    args = parser.parse_args()

    extractor = PDFImageExtractor(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        tracing_enabled=args.tracing_enabled,
        tracing_dir=args.tracing_dir,
        use_pymupdf_for_caption =args.use_pymupdf_for_caption
    )
    images_dict, captions_dict = extractor.process_pdf()
    print("captions_dict", captions_dict)


if __name__ == "__main__":
    main()
