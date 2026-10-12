# TODO - Greeting rules + PDF fallback

- [x] Update greeting template to match required Greeting Rules message.
- [x] Ensure greeting intent routes to greeting_template.
- [x] Update PDFLoader to try PyMuPDF (fitz) -> pdfplumber -> PyPDF2, with user-friendly missing-module messaging.
- [x] Update DocumentIngestion to use PDFLoader fallback outputs and avoid exposing raw exceptions.
- [x] Ensure UI upload page does not expose raw Python exceptions.

