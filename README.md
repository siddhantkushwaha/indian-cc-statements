# Indian Bank Statements Parser

A Python package for extracting transaction data from Indian bank credit card statement PDFs using OCR and AI-powered table detection.

## Features

- 🔓 **PDF Unlocking**: Automatically unlock password-protected PDFs
- 🤖 **AI Table Detection**: Uses Microsoft's Table Transformer model to detect tables
- 📝 **OCR Extraction**: Extracts text using EasyOCR with high accuracy
- 💰 **Transaction Parsing**: Intelligently parses dates, amounts, and merchant names
- 📊 **Multiple Formats**: Exports to JSON and CSV

## Installation

### Prerequisites

**macOS:**
```bash
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
Download poppler from [here](https://github.com/oschwartz10612/poppler-windows/releases/) and add to PATH.

### Install Package

```bash
pip install indian-cc-statements
```

Or install from source:

```bash
git clone https://github.com/siddhantkushwaha/indian-cc-statements.git
cd indian-cc-statements
pip install -e .
```

```
pip install git+https://github.com/siddhantkushwaha/indian-cc-statements.git
```

## Usage

### Command Line

```bash
# Single PDF
indian-cc-statements --pdf-path statement.pdf

# Multiple PDFs
indian-cc-statements --pdf-path statement1.pdf,statement2.pdf

# Password-protected PDFs
indian-cc-statements --pdf-path statement.pdf --password password123,password456
```

### Python API

```python
from indian_cc_statements import extract

# Extract transactions from PDF
transactions = extract(
    pdf_path="statement.pdf",
    expand_x=0.15,               # Horizontal table expansion (15%)
    expand_y=0.10,               # Vertical table expansion (10%)
    confidence_threshold=0.5,    # Table detection confidence
    passwords=["password123"]    # Optional: for encrypted PDFs
)

# Process results
for txn in transactions:
    print(f"Date: {txn['Date']}")
    print(f"Merchant: {txn['Merchant']}")
    print(f"Amount: {txn['Amount']}")
    print(f"Type: {txn['Credit/Debit']}")
    print("---")
```

## Output Format

The package extracts transactions in the following format:

```json
[
    {
        "Date": "01/12/2024",
        "Merchant": "Amazon Prime",
        "Amount": 1234.56,
        "Credit/Debit": "Debit"
    },
    {
        "Date": "05/12/2024",
        "Merchant": "Salary Credit",
        "Amount": 50000.00,
        "Credit/Debit": "Credit"
    }
]
```

## How It Works

The extraction process follows a hierarchical approach:

1. **PDF Level**: Unlock if needed, convert to high-res images
2. **Page Level**: Detect all tables using Table Transformer AI model
3. **Table Level**: Crop tables, run OCR, validate transaction data
4. **Line Level**: Parse dates/amounts, identify merchant columns

## Configuration

### Key Parameters

- `expand_x` (float): Horizontal padding around detected tables (default: 0.15)
- `expand_y` (float): Vertical padding around detected tables (default: 0.10)
- `confidence_threshold` (float): Minimum confidence for table detection (default: 0.5)
- `passwords` (list): Passwords to try for encrypted PDFs

## Requirements

- Python 3.8+
- poppler (for PDF rendering)
- GPU recommended for faster processing (CUDA or Apple MPS)

## Limitations

- Designed for Indian bank credit card statements
- Requires clear, readable PDFs (scanned documents may have lower accuracy)
- Transaction tables must follow standard formats (Date, Description, Amount columns)

## Development

```bash
# Clone repository
git clone https://github.com/siddhantkushwaha/indian-cc-statements.git
cd indian-cc-statements

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on [GitHub](https://github.com/siddhantkushwaha/indian-cc-statements/issues).

