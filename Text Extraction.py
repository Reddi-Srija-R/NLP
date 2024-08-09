import os  
import io
import numpy as np
import logging

import cv2
import fitz
import pytesseract
from docx import Document
from PIL import Image, ImageFile
import logging
 
logging.basicConfig(filename='text_extraction.log', level=logging.INFO)

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Tesseract\tesseract.exe'
 
class TextImageExtractor:
    def __init__(self):
        pass
 
    def extract_text_and_images(self, file_path):
        """Extract text and images with descriptions from a file."""
        text = ""
        images_with_text = []
 
        try:
            if file_path.lower().endswith('.pdf'):
                text, images_with_text = self.extract_text_and_images_from_pdf(file_path)
                logging.info("Extracted text and images (including text) from PDF file.")

            elif file_path.lower().endswith('.docx'):
                text, images_with_text = self.extract_text_and_images_from_docx(file_path)
                logging.info("Extracted text and images (including text) from DOCX file.")

            elif file_path.lower().endswith('.txt'):
                text = self.extract_text_from_txt(file_path)
                logging.info("Extracted text from TXT file.")

            elif any(file_path.lower().endswith(image_ext) for image_ext in ['.jpg', '.jpeg', '.png', '.gif']):
                text = self.preprocess_and_extract_text(file_path)
                images_with_text = [(Image.open(file_path), text)]
                logging.info("Extracted text from image files.")

            else:
                logging.error(f"Unsupported file format: {file_path}")

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
 
        # Filter out images with empty or None text
        images_with_text = [(img, img_text) for img, img_text in images_with_text if img_text.strip()]
 
        return text, images_with_text
 
    def extract_text_and_images_from_pdf(self, pdf_path):
        """Extract text and images with descriptions from a PDF file."""
        text = ""
        images_with_text = []
 
        try:
            doc = fitz.open(pdf_path)
 
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text
 
                # Extract images with their text descriptions
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
 
                    # OCR to extract text from image
                    image_text = self.extract_text_from_image(image)
 
                    # Add to list only if text is not empty or None
                    if image_text.strip():
                        images_with_text.append((image, image_text))
 
            return text, images_with_text
 
        except Exception as e:
            logging.error(f"Error extracting text and images from PDF {pdf_path}: {str(e)}")
            return "", []
 
    def extract_text_and_images_from_docx(self, docx_path):
        """Extract text and images with descriptions from a DOCX file."""
        text = ""
        images_with_text = []
 
        try:
            doc = Document(docx_path)
 
            for paragraph in doc.paragraphs:
                paragraph_text = paragraph.text + "\n"
                text += paragraph_text
 
                # Extract images with their text descriptions
                for run in paragraph.runs:
                    if run._element.tag == '{http://schemas.openxmlformats.org/drawingml/2006/picture}pic':
                        image = run.inline_shapes[0].image
                        image_bytes = image.blob
                        image = Image.open(io.BytesIO(image_bytes))
 
                        # OCR to extract text from image
                        image_text = self.extract_text_from_image(image)
 
                        # Add to list only if text is not empty or None
                        if image_text.strip():
                            images_with_text.append((image, image_text))
 
            return text, images_with_text
 
        except Exception as e:
            logging.error(f"Error extracting text and images from DOCX {docx_path}: {str(e)}")
            return "", []
 
    def extract_text_from_txt(self, txt_path):
        """Extract text from a plain TXT file."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text
 
        except Exception as e:
            logging.error(f"Error extracting text from TXT {txt_path}: {str(e)}")
            return ""
 
    def preprocess_and_extract_text(self, image_path):
        """Preprocess image and extract text using OCR."""
        try:
            # Load image
            image =  np.array(image_path)
 
            # Correct skewness
            skew_corrected_image, skew_angle = self.correct_skew(image)
 
            # Sharpen the image to reduce blurriness
            sharpened_image = self.sharpen_image(skew_corrected_image)
 
            # Convert sharpened image to grayscale
            gray_sharpened = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
 
            # Apply Gaussian blur to reduce noise
            blurred_corrected = cv2.GaussianBlur(gray_sharpened, (5, 5), 0)
 
            # Perform adaptive thresholding to create binary image
            _, binary_corrected = cv2.threshold(blurred_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
            # Invert binary image (if needed)
            binary_corrected = cv2.bitwise_not(binary_corrected)
 
            # Perform OCR using pytesseract on the sharpened image
            extracted_text = pytesseract.image_to_string(Image.fromarray(binary_corrected))

            return extracted_text.strip()

        # If image file doesn't exist
        except FileNotFoundError as fe:
            logging.error(f"File not found error: {fe}")

        # Cathes errors raised by OpenCV functions during image processing
        except cv2.error as cve:
            logging.error(f"OpenCV error during image preprocessing: {cve}")

        except pytesseract.TesseractError as tse:
            logging.error(f"Tesseract OCR error: {tse}")
 
    def sharpen_image(self, image):
        """Sharpen the image using Unsharp Masking."""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return sharpened
 
    def correct_skew(self, image):
        """Correct skewness in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
        # Detect lines in the image using Hough transform
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
 
        # Calculate the median angle (most prominent direction of lines)
        median_angle = np.median(angles)
 
        # Rotate the image to correct skew
        rotated = self.rotate_image(image, median_angle)
 
        return rotated, median_angle
 
    def rotate_image(self, image, angle):
        """Rotate the image by the specified angle."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
 
    def extract_text_from_image(self, image):
        """Extract text from an image using OCR."""
        try:
            # Preprocess image if necessary
            processed_image = self.preprocess_and_extract_text(image)

            # ImageFile.LOAD_TRUNCATED_IMAGES = True
            # extracted_text = self.pytesseract.image_to_string(image)
 
            return processed_image

        except Exception as e:
            logging.error(f"Error extracting text from image: {str(e)}")
            return ""
 
    def save_text_to_file(self, text, images_with_text, file_path):
        """Save extracted text (including text from images) to a text file."""
        try:
            file_name = os.path.basename(file_path)
            output_folder = 'Extracted Output Folder'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
 
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")
 
            filtered_images_with_text = [(img, img_text) for img, img_text in images_with_text if img_text.strip()]
 
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(f"Main Text:\n{text}\n\n")
                file.write("Text from Images:\n")
                for img, img_text in filtered_images_with_text:
                    file.write(f"Image: {img}\n")
                    file.write(f"{img_text}\n\n")
 
            logging.info(f"Successfully saved text to {output_file_path}")
 
        except Exception as e:
            logging.error(f"Error saving text to file {output_file_path}: {str(e)}")
 
if __name__ == "__main__":

    folder_path = 'Input Data'
 
    extractor = TextImageExtractor()
 
    try:
 
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)

                text, images_with_text = extractor.extract_text_and_images(file_path)
                images_text = list(filter(lambda x: len(x) > 0, images_with_text))

                extractor.save_text_to_file(text, images_text, file_path)
                
                # Print filename and number of images extracted
                print(f"File: {os.path.basename(file_path)}\n")
                # if text != ' ':
                #     print(text[:500])
                # print(f"Number of Images Extracted: {len(images_with_text)}")
                # if len(images_text) != 0:
                #     images_text = images_text[0].replace('\n\n',' ').replace('\n',' ')
                #     print(images_text)
                # print()
                             
    except FileNotFoundError as F:
        logging.error(f"The folder path '{folder_path}' does not exist.")
    
    except Exception as e:
        logging.error(e)