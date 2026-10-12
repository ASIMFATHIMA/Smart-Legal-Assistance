from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional


def _format_missing_module(module_name: str) -> str:
    return (
        "🔍 Error Detected\n\n"
        "Missing Module:\n"
        f"{module_name}\n\n"
        "Solution:\n"
        f"python -m pip install {module_name}\n\n"
        "Verification:\n"
        f"python -c \"import {module_name}; print('Installed')\"\n"
    )


class PDFLoader:
    def __init__(self, fallback_ocr: bool = True):
        self.fallback_ocr = fallback_ocr

    def _print_debug_info(self, pages_text: List[Dict[str, str]], method_name: str) -> None:
        total_pages = len(pages_text)
        total_chars = sum(len(p.get("text", "")) for p in pages_text)
        first_500 = ""
        if pages_text:
            first_500 = pages_text[0].get("text", "")[:500]
        print(f"--- PDF Extraction Debug ({method_name}) ---")
        print(f"Number of Pages: {total_pages}")
        print(f"Extracted Characters: {total_chars}")
        print(f"First 500 characters:\n{first_500}")
        print("------------------------------------------")

    def extract_text(self, file_path: str) -> List[Dict[str, str]]:
        """Extracts text from a PDF file.

        Fallback chain:
          1) PyMuPDF (fitz)
          2) pdfplumber
          3) PyPDF2

        Returns:
          List[{'page_number': int, 'text': str}]
        """

        # 1) Try PyMuPDF (fitz)
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            pages_text: List[Dict[str, str]] = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()

                if (not text or len(text.strip()) < 50) and self.fallback_ocr:
                    try:
                        from src.ocr_handler import OCRHandler
                        import os
                        
                        # Find Tesseract path in common Windows installation paths
                        possible_paths = [
                            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                            os.environ.get("TESSERACT_PATH", "")
                        ]
                        tess_path = None
                        for p in possible_paths:
                            if p and os.path.exists(p):
                                tess_path = p
                                break
                        
                        ocr = OCRHandler(tesseract_path=tess_path)
                        
                        # Convert page to high-quality image (2x zoom)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        import io
                        from PIL import Image
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        
                        # Preprocess image for OCR
                        img = ocr.preprocess_image_for_ocr(img)
                        
                        # Extract text
                        import pytesseract
                        text = pytesseract.image_to_string(img, lang=ocr.language).strip()
                    except Exception as e:
                        # OCR is optional; ignore OCR errors and keep going
                        print(f"[OCR fallback] Warning: OCR page extraction failed: {e}")
                        pass

                if text:
                    pages_text.append({"page_number": page_num + 1, "text": text})

            doc.close()
            self._print_debug_info(pages_text, "PyMuPDF")
            return pages_text

        except ModuleNotFoundError as e:
            if e.name and e.name.lower() in {"fitz", "pymupdf"}:
                # fitz isn't available: continue to pdfplumber fallback
                pass
            else:
                raise

        except Exception:
            # Extraction failed: continue with pdfplumber fallback
            pass

        # 2) Try pdfplumber
        try:
            import pdfplumber

            pages_text = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        pages_text.append({"page_number": i, "text": text})
            self._print_debug_info(pages_text, "pdfplumber")
            return pages_text
        except ModuleNotFoundError as e:
            if e.name:
                # pdfplumber missing: continue to PyPDF2 fallback
                pass
        except Exception:
            pass

        # 3) Try PyPDF2
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            pages_text = []
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages_text.append({"page_number": i, "text": text})
            self._print_debug_info(pages_text, "PyPDF2")
            return pages_text
        except ModuleNotFoundError as e:
            # Provide a user-friendly error about the missing dependency
            if e.name:
                err_text = _format_missing_module(e.name)
                res = [{"page_number": 1, "text": err_text}]
                self._print_debug_info(res, "Fallback Error")
                return res
        except Exception:
            # Last resort: no text extracted
            self._print_debug_info([], "Failed")
            return []

