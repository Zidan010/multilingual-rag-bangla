import pytesseract
from pdf2image import convert_from_path
import os

# path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# path to poppler bin directory
poppler_path = r"C:\Release-24.07.0-0\poppler-24.07.0\Library\bin"  

pdf_path = "HSC26-Bangla1st-Paper.pdf"
first_page = 6
last_page = 19

# Convert PDF pages to images
print("Converting PDF pages to images...")
pages = convert_from_path(
    pdf_path,
    dpi=300,
    first_page=first_page,
    last_page=last_page,
    poppler_path=poppler_path  
)

# Create output folder for images
os.makedirs("pages_images", exist_ok=True)

# OCR all pages
output_text = ""
for i, img in enumerate(pages):
    page_number = i + first_page
    img_path = f"all_pages_images/page_{page_number}.png"
    img.save(img_path)

    print(f"OCR on page {page_number}...")
    text = pytesseract.image_to_string(img, lang='ben')  # Bengali OCR
    output_text += f"\n\n--- Page {page_number} ---\n{text}"

# Save final output
output_file = "bangla_ocr_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"\nOCR complete. Output saved to '{output_file}'")
