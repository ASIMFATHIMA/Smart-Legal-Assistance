"""
Smart Legal Assistance - src package
====================================
This package contains the core modules for legal PDF processing.
"""

from .preprocessing import Preprocessor

# Optional import: pdf_extractor requires PyMuPDF (fitz).
# Keep package import working even when fitz isn't installed.
try:
    from .pdf_extractor import PDFExtractor, get_all_pdfs_from_folder
except ModuleNotFoundError:
    PDFExtractor = None
    get_all_pdfs_from_folder = None

__all__ = [
    'PDFExtractor',
    'Preprocessor',
    'get_all_pdfs_from_folder'
]


__version__ = '1.0.0'
__author__ = 'Senior NLP Engineer'
__description__ = 'Smart Legal Assistance - Phase 1 PDF Processing'
