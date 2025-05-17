import os
import fitz  # PyMuPDF
import tabula
from pathlib import Path

PDF_DIR = "data/pdf_reports"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_tables_from_pdf(pdf_path, output_dir):
    """Extract tables from PDF and save as CSV."""
    tables = tabula.read_pdf(pdf_path, pages="all")
    for idx, table in enumerate(tables):
        table_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_table_{idx}.csv")
        table.to_csv(table_path, index=False)
    doc = fitz.open(pdf_path)

def extract_images_from_pdf(pdf_path, output_dir):
    """Extract images from PDF and save as PNG."""
    for i in range(len(doc)):
        page = doc[i]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_page_{i}_img_{img_index}.png")
            with open(image_path, "wb") as img_file:
                img_file.write(base_image["image"])

def process_pdf(pdf_path):
    """Process a single PDF and save extracted data."""
    output_dir = os.path.join(OUTPUT_DIR, Path(pdf_path).stem)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    with open(os.path.join(output_dir, "text.txt"), "w") as text_file:
        text_file.write(text)
    
    # Extract tables
    extract_tables_from_pdf(pdf_path, output_dir)
    
    # Extract images
    extract_images_from_pdf(pdf_path, output_dir)

def main():
    pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        process_pdf(pdf_file)

if __name__ == "__main__":
    main()
