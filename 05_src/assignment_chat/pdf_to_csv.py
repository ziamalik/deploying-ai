import os
import csv
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path} with pypdf: {e}")
    return text.strip()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    slides_dir = os.path.join(base_dir, "01_materials", "slides")
    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
    
    # Check if slides dir exists
    if not os.path.exists(slides_dir):
        print(f"Directory {slides_dir} not found.")
        return

    pdf_files = [f for f in os.listdir(slides_dir) if f.endswith(".pdf")]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source_file', 'content'])
        
        for index, pdf_file in enumerate(pdf_files):
            print(f"Processing {index+1}/{len(pdf_files)}: {pdf_file}")
            pdf_path = os.path.join(slides_dir, pdf_file)
            
            # Since some slides have huge text blocks, we can just split by page or chunk later.
            # For simplicity, let's extract all text per slide deck, and chunk it later in the RAG pipeline.
            text = extract_text_from_pdf(pdf_path)
            if text:
                # Clean up multiple newlines or weird spacing if needed, but keeping it raw is fine for embeddings
                writer.writerow([pdf_file, text])
            else:
                print(f"Warning: No text extracted from {pdf_file}")
                
    print(f"Successfully generated {output_csv}")

if __name__ == "__main__":
    main()
