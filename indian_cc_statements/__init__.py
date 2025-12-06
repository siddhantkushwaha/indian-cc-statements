"""
Indian Credit Card Statements Parser

A Python package for extracting transaction data from Indian credit card statement PDFs.
Uses OCR and AI-powered table detection to parse transaction tables.
"""

__version__ = "0.1.0"
__author__ = "Siddhant Kushwaha"

from .parser import extract

__all__ = ["extract"]
