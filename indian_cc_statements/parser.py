"""
Extracts transaction as JSON for credit card statement PDFs.

HIGH-LEVEL APPROACH:
1. PDF Level: Unlock if needed, convert to images, initialize OCR, process each page
2. Page Level: Detect all tables on page, process each table
3. Table Level: Crop table, run OCR, validate transaction data, save artifacts
4. Line Level: Group OCR text by lines, parse dates/amounts, identify merchant column

Requirements:
- poppler must be installed for pdf2image: brew install poppler (on macOS)
- Ollama must be running locally with a model (e.g., llama3.2)
- Table Transformer model for table detection
- pikepdf for handling encrypted PDFs
"""

import os
import json
import re
import argparse
import tempfile
import warnings
from dateutil import parser as date_parser
from typing import List, Optional, Tuple
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_path
import easyocr
import torch

# Suppress urllib3 OpenSSL warning
warnings.filterwarnings("ignore", category=Warning, module="urllib3")
# Suppress PyTorch MPS pin_memory warning
warnings.filterwarnings("ignore", message=".*pin_memory.*")

from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import pikepdf
import pandas as pd


# ============================================================================
# TABLE DETECTION MODEL
# ============================================================================


class TableDetector:
    """Singleton for Microsoft's Table Transformer (DETR-based) model."""

    _instance = None
    _processor = None
    _model = None

    @classmethod
    def get_instance(cls) -> "TableDetector":
        """Get singleton instance of TableDetector."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Load Table Transformer model once."""
        if TableDetector._model is None:
            print("Loading Table Transformer model...")
            TableDetector._processor = DetrImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection", revision="no_timm"
            )
            TableDetector._model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection", revision="no_timm"
            )
            # Use GPU/MPS if available
            if torch.cuda.is_available():
                TableDetector._model = TableDetector._model.cuda()
            elif torch.backends.mps.is_available():
                TableDetector._model = TableDetector._model.to("mps")
            print("Table Transformer model loaded.")

    def detect(
        self, image: Image.Image, confidence_threshold: float = 0.7
    ) -> List[dict]:
        """Detect tables in image, return list of bbox + confidence."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess and move to model device
        inputs = self._processor(images=image, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process to get bounding boxes
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self._processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )[0]

        tables = []
        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            tables.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(score),
                    "label": int(label),
                }
            )

        return tables


# ============================================================================
# PDF LEVEL: Unlock encrypted PDFs
# ============================================================================


def is_pdf_encrypted(pdf_path: str) -> bool:
    """Check if PDF is encrypted/password protected."""
    try:
        with pikepdf.open(pdf_path) as pdf:
            return False
    except pikepdf.PasswordError:
        return True
    except Exception as e:
        print(f"Error checking PDF encryption: {e}")
        return False


def unlock_pdf(pdf_path: str, passwords: List[str]) -> Optional[str]:
    """
    Try to unlock PDF with password list.

    Returns:
        Path to unlocked PDF (temp file) if successful, None if all passwords fail
    """
    if not passwords:
        return None

    for password in passwords:
        try:
            # Try opening with this password
            with pikepdf.open(pdf_path, password=password) as pdf:
                # Success! Save unlocked version to temp file
                temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")
                os.close(temp_fd)
                pdf.save(temp_path)
                print(f"Successfully unlocked PDF with password: {'*' * len(password)}")
                return temp_path
        except pikepdf.PasswordError:
            continue
        except Exception as e:
            print(f"Error trying password: {e}")
            continue

    print(f"Failed to unlock PDF - none of the {len(passwords)} passwords worked")
    return None


def prepare_pdf(pdf_path: str, passwords: Optional[List[str]] = None) -> Optional[str]:
    """
    Prepare PDF for processing - unlock if needed.

    Returns:
        Path to usable PDF (original if unlocked, temp file if was locked), None if can't unlock
    """
    # Check if PDF is encrypted
    if not is_pdf_encrypted(pdf_path):
        print(f"PDF is not encrypted, using original: {pdf_path}")
        return pdf_path

    print(f"PDF is encrypted, attempting to unlock...")

    # No passwords provided
    if not passwords:
        print("No passwords provided for encrypted PDF")
        return None

    # Try to unlock with password list
    return unlock_pdf(pdf_path, passwords)


# ============================================================================
# PDF LEVEL: Convert PDF to images
# ============================================================================


def pdf_to_images(pdf_path: str, dpi: int = 500) -> List[Image.Image]:
    """Convert PDF pages to high-resolution images."""
    print(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"Converted {len(images)} pages to images")
    return images


# ============================================================================
# PAGE LEVEL: Detect tables on a page
# ============================================================================


def detect_tables_on_page(
    image: Image.Image,
    confidence_threshold: float,
    expand_x: float = 0.15,
    expand_y: float = 0.10,
) -> List[dict]:
    """Detect and expand table regions on a single page."""
    detector = TableDetector.get_instance()
    print("Detecting tables with Table Transformer...")
    detections = detector.detect(image, confidence_threshold=confidence_threshold)

    if not detections:
        print("No tables detected")
        return []

    img_width, img_height = image.size
    table_regions = []

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        table_width = x2 - x1
        table_height = y2 - y1

        # Expand bbox by percentage of table size
        pad_x = int(table_width * expand_x)
        pad_y = int(table_height * expand_y)

        # Clamp to image bounds
        x1_expanded = max(0, x1 - pad_x)
        y1_expanded = max(0, y1 - pad_y)
        x2_expanded = min(img_width, x2 + pad_x)
        y2_expanded = min(img_height, y2 + pad_y)

        expanded_bbox = (x1_expanded, y1_expanded, x2_expanded, y2_expanded)

        table_regions.append(
            {
                "bbox": expanded_bbox,
                "bbox_original": det["bbox"],
                "page": 0,
                "table_index": i,
                "confidence": det["confidence"],
            }
        )
        print(
            f"  Table {i}: bbox={expanded_bbox} (expanded {expand_x*100:.0f}%x, {expand_y*100:.0f}%y from {det['bbox']}), confidence={det['confidence']:.2f}"
        )

    print(f"Found {len(table_regions)} table(s)")
    return table_regions


# ============================================================================
# TABLE LEVEL: Crop table region
# ============================================================================


def crop_table_region(image: Image.Image, bbox: tuple) -> Image.Image:
    """Crop table region from image using bbox."""
    x1, y1, x2, y2 = bbox
    width, height = image.size

    # Clamp coordinates to image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))

    return image.crop((x1, y1, x2, y2))


# ============================================================================
# TABLE LEVEL: OCR - Extract text grouped by lines
# ============================================================================


def run_ocr_on_table(
    image: Image.Image,
    reader: easyocr.Reader,
    min_confidence: float = 0.4,
    y_tolerance: int = 15,
    x_tolerance: int = 40,
) -> List[List[str]]:
    """
    Run OCR and group text into lines and sentences.

    Returns list of lines, each line is list of sentences:
    [["Date", "Description", "Amount"], ["01/12/2024", "Amazon Prime", "1,234.56"], ...]
    """
    # Convert PIL to numpy array for EasyOCR
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Perform OCR
    raw_results = reader.readtext(img_array)

    # Filter by confidence and extract positions
    results = []
    for bbox, text, confidence in raw_results:
        if confidence >= min_confidence:
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            left_x = bbox[0][0]
            right_x = bbox[1][0]
            results.append((text, center_y, left_x, right_x))
        else:
            print(f"Skipping text with confidence {text} {confidence}")

    if not results:
        return []

    # Group into lines by y-coordinate
    results.sort(key=lambda r: r[1])
    raw_lines = _group_text_into_lines(results, y_tolerance)

    # Merge adjacent text within each line
    lines = []
    for raw_line in raw_lines:
        sentences = _merge_adjacent_text(raw_line, x_tolerance)
        lines.append(sentences)

    return lines


def _group_text_into_lines(
    results: List[Tuple[str, float, float, float]], y_tolerance: int
) -> List[List[Tuple[str, float, float, float]]]:
    """Group OCR results into lines based on y-coordinate proximity."""
    raw_lines = []
    current_line = [results[0]]
    current_y = results[0][1]

    for result in results[1:]:
        if abs(result[1] - current_y) <= y_tolerance:
            current_line.append(result)
        else:
            raw_lines.append(current_line)
            current_line = [result]
            current_y = result[1]

    if current_line:
        raw_lines.append(current_line)

    return raw_lines


def _merge_adjacent_text(
    raw_line: List[Tuple[str, float, float, float]], x_tolerance: int
) -> List[str]:
    """Merge adjacent text nodes within x_tolerance into sentences."""
    raw_line.sort(key=lambda r: r[2])  # Sort by left_x

    sentences = []
    current_sentence = raw_line[0][0]
    current_right_x = raw_line[0][3]

    for item in raw_line[1:]:
        text, center_y, left_x, right_x = item
        gap = left_x - current_right_x

        if gap <= x_tolerance:
            current_sentence += " " + text
            current_right_x = right_x
        else:
            sentences.append(current_sentence)
            current_sentence = text
            current_right_x = right_x

    sentences.append(current_sentence)
    return sentences


# ============================================================================
# LINE LEVEL: Validate transaction data
# ============================================================================


def check_transaction_table(
    ocr_lines: List[List[str]], min_dates: int = 1, min_amounts: int = 1
) -> Tuple[bool, dict]:
    """Check if OCR lines represent a transaction table."""
    # Find most common column count
    most_common_column_count = _find_most_common_column_count(ocr_lines)

    # Parse each line for date, amount, merchant
    data, useless_lines = _parse_transaction_lines(ocr_lines, most_common_column_count)

    # Identify merchant column (longest average text)
    avg_column_length = _calculate_avg_column_lengths(data, most_common_column_count)
    merchant_column_idx = (
        max(range(len(avg_column_length)), key=avg_column_length.__getitem__)
        if avg_column_length
        else 0
    )

    # Add merchant to each transaction
    for line_dict in data:
        line_dict["Merchant"] = line_dict["_raw_text"][merchant_column_idx]
        del line_dict["_raw_text"]

    # Determine if valid transaction table
    total_lines = len(useless_lines) + len(data)
    percentage_useless_lines = (
        len(useless_lines) / total_lines if total_lines > 0 else 0
    )
    is_txn_table = len(data) >= 1 and (
        len(useless_lines) <= 3 or percentage_useless_lines <= 0.1
    )

    details = {
        "total_lines": total_lines,
        "lines_with_data": len(data),
        "data": data,
        "useless_lines": useless_lines,
        "percentage_useless_lines": percentage_useless_lines,
        "most_common_column_count": most_common_column_count,
    }
    return is_txn_table, details


def _find_most_common_column_count(ocr_lines: List[List[str]]) -> int:
    """Find most frequently occurring column count (lines with 3+ columns)."""
    column_counts = {}
    for line in ocr_lines:
        if len(line) < 3:
            continue
        line_length = len(line)
        column_counts[line_length] = column_counts.get(line_length, 0) + 1
    return max(column_counts, key=column_counts.get) if column_counts else 0


def _parse_transaction_lines(
    ocr_lines: List[List[str]], expected_column_count: int
) -> Tuple[List[dict], List[list]]:
    """Parse lines to extract date, amount, and identify valid transactions."""
    data = []
    useless_lines = []

    # Compile amount regex patterns
    amount_with_suffix = re.compile(
        r"\b\d{1,3}(?:,\d{2,3})*\.\d{2}\s*(?:Dr|Cr)\b", re.IGNORECASE
    )
    amount_plain = re.compile(r"\b\d{1,3}(?:,\d{2,3})*\.\d{2}\b")

    for line in ocr_lines:
        if len(line) != expected_column_count:
            continue

        line_date, line_amount, invalid = _extract_date_and_amount(
            line, amount_with_suffix, amount_plain
        )

        if not invalid and line_date and line_amount:
            line_amount = line_amount.lower()
            is_amounted_credit = "cr" in line_amount
            line_amount = (
                line_amount.replace("cr", "").replace("dr", "").replace(",", "").strip()
            )
            data.append(
                {
                    "_raw_text": line,
                    "Date": line_date,
                    "Amount": float(line_amount),
                    "Credit/Debit": "Credit" if is_amounted_credit else "Debit",
                }
            )
        else:
            useless_lines.append([line, invalid, line_date, line_amount])

    return data, useless_lines


def _extract_date_and_amount(
    line: List[str], amount_with_suffix: re.Pattern, amount_plain: re.Pattern
) -> Tuple[Optional[str], Optional[str], bool]:
    """Extract date and amount from a single line, return (date, amount, invalid)."""
    line_date = None
    line_amount = None
    invalid = False

    for text in line:
        # Check for date (max 17 chars for dates like "12 September 2024")
        if _is_valid_date(text) and len(text) <= 17:
            if not line_date:
                line_date = text
            else:
                invalid = True
                break

        # Check for amount (with or without Dr/Cr suffix)
        new_amount = _extract_amount(text, amount_with_suffix, amount_plain)
        if new_amount:
            if not line_amount:
                line_amount = new_amount
            else:
                invalid = True
                break

    return line_date, line_amount, invalid


def _is_valid_date(text: str) -> bool:
    """Check if text is a valid date using dateutil parser."""
    try:
        if text.isdigit() or len(text) < 6:
            return False
        parsed = date_parser.parse(text, fuzzy=False)
        return 1990 <= parsed.year <= 2100
    except (ValueError, TypeError, OverflowError):
        return False


def _extract_amount(
    text: str, amount_with_suffix: re.Pattern, amount_plain: re.Pattern
) -> Optional[str]:
    """Extract amount if text matches amount pattern (with/without Dr/Cr suffix)."""
    # First check for Dr/Cr suffix
    suffix_matches = amount_with_suffix.findall(text)
    if len(suffix_matches) > 0 and len(text) == len(suffix_matches[0]):
        return text

    # Fall back to plain amounts
    plain_matches = amount_plain.findall(text)
    if len(plain_matches) > 0 and len(text) == len(plain_matches[0]):
        return text

    return None


def _calculate_avg_column_lengths(data: List[dict], column_count: int) -> List[float]:
    """Calculate average text length for each column to identify merchant column."""
    avg_column_length = [0] * column_count

    for line_dict in data:
        for idx, text in enumerate(line_dict["_raw_text"]):
            avg_column_length[idx] += len(text)

    return [length / len(data) if len(data) > 0 else 0 for length in avg_column_length]


# ============================================================================
# TABLE LEVEL: Save table artifacts
# ============================================================================


def save_table_artifacts(
    output_dir: str,
    page_num: int,
    table_idx: int,
    is_txn: bool,
    details: dict,
    ocr_lines: List[List[str]],
    cropped_image: Image.Image,
):
    """Save table metadata (JSON), text, and image to disk."""
    table_identifier = f"page_{page_num + 1}_table_{table_idx}_{is_txn}"

    # Save metadata JSON
    json_path = os.path.join(output_dir, f"{table_identifier}.json")
    with open(json_path, "w") as f:
        json.dump(details, f, indent=4)

    # Save text
    table_text = "\n".join([" | ".join(line) for line in ocr_lines])
    text_path = os.path.join(output_dir, f"{table_identifier}.txt")
    with open(text_path, "w") as f:
        f.write(table_text)

    # Save image
    image_path = os.path.join(output_dir, f"{table_identifier}.png")
    cropped_image.save(image_path)


# ============================================================================
# TABLE LEVEL: Process single table
# ============================================================================


def process_table(
    image: Image.Image,
    table_info: dict,
    page_num: int,
    reader: easyocr.Reader,
    output_dir: str,
) -> List[dict]:
    """Process single table: crop, OCR, validate, save artifacts."""
    bbox = table_info["bbox"]
    table_idx = table_info["table_index"]

    # Crop table region
    cropped_image = crop_table_region(image, bbox)

    # Run OCR to get lines
    ocr_lines = run_ocr_on_table(cropped_image, reader)

    # Check if transaction table
    is_txn, details = check_transaction_table(ocr_lines)

    # Save artifacts for inspection
    save_table_artifacts(
        output_dir, page_num, table_idx, is_txn, details, ocr_lines, cropped_image
    )

    return details["data"]


# ============================================================================
# PAGE LEVEL: Process all tables on a page
# ============================================================================


def process_page(
    image: Image.Image,
    page_num: int,
    reader: easyocr.Reader,
    output_dir: str,
    confidence_threshold: float,
    expand_x: float,
    expand_y: float,
) -> List[dict]:
    """Process single page: detect tables, process each table."""
    print(f"\nProcessing page {page_num + 1}")

    # Detect tables on page
    table_regions = detect_tables_on_page(
        image,
        confidence_threshold=confidence_threshold,
        expand_x=expand_x,
        expand_y=expand_y,
    )
    print(f"Found {len(table_regions)} table(s) on page {page_num + 1}")

    if not table_regions:
        print(f"No tables detected on page {page_num + 1}, skipping...")
        return []

    # Process each table on page
    page_data = []
    for table_info in table_regions:
        table_data = process_table(image, table_info, page_num, reader, output_dir)
        page_data.extend(table_data)

    return page_data


# ============================================================================
# PDF LEVEL: Main extraction orchestrator
# ============================================================================


def extract(
    pdf_path: str,
    expand_x: float = 0.15,
    expand_y: float = 0.10,
    confidence_threshold: float = 0.5,
    passwords: Optional[List[str]] = None,
):
    """
    Extract transaction data from PDF credit card statement.

    Args:
        pdf_path: Path to PDF file
        expand_x: Horizontal expansion fraction (0.15 = 15%)
        expand_y: Vertical expansion fraction (0.10 = 10%)
        confidence_threshold: Table detection confidence (0.0 to 1.0)
        passwords: List of passwords to try if PDF is encrypted (optional)

    Returns:
        List of all extracted transactions
    """

    # Prepare PDF - unlock if needed
    usable_pdf_path = prepare_pdf(pdf_path, passwords)
    if not usable_pdf_path:
        print(f"Cannot process PDF - failed to unlock: {pdf_path}")
        return []

    try:
        # Convert PDF to images
        images = pdf_to_images(usable_pdf_path)

        # Initialize OCR reader once
        print("Initializing EasyOCR reader...")
        reader = easyocr.Reader(["en"], gpu=True)

        # Process each page
        all_data = []
        output_dir = tempfile.mkdtemp(prefix="pdf_extract_")
        for page_num, image in enumerate(images):
            page_data = process_page(
                image,
                page_num,
                reader,
                output_dir,
                confidence_threshold,
                expand_x,
                expand_y,
            )
            all_data.extend(page_data)

        return all_data
    except Exception as e:
        print(f"Error extracting transactions from PDF: {e}")
        return []
