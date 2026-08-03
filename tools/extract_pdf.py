import sys
import os

pdf_path = sys.argv[1]

try:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    with open("paper_thermal_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Extracted to paper_thermal_text.txt via PyMuPDF")
except ImportError:
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        with open("paper_thermal_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted to paper_thermal_text.txt via PyPDF2")
    except Exception as e:
        print(f"Failed via PyPDF2 as well: {e}")
except Exception as e:
    print(f"Failed: {e}")
