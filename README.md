# AI PDF Translator

Translate English PDFs to French **in-place**, preserving the original layout, images, fonts and colors.

Uses [PyMuPDF](https://pymupdf.readthedocs.io/) to redact and rewrite text directly in the PDF.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python translate_pdf.py document.pdf
```

Output: `document_fr.pdf`

### Options

| Flag | Description |
|---|---|
| `-o`, `--output` | Custom output path |
| `-w`, `--workers` | Number of parallel translation threads (default: 10) |
| `-e`, `--engine` | Translation engine: `google` or `openai` (default: google) |

### Translation engines

**Google Translate** (default) — Free, no API key required.

```bash
python translate_pdf.py document.pdf
```

**OpenAI** — Uses `gpt-4o-mini`. Requires the `OPENAI_API_KEY` environment variable.

```bash
export OPENAI_API_KEY=sk-...
python translate_pdf.py document.pdf -e openai
```

### Examples

```bash
# Custom output path
python translate_pdf.py document.pdf -o translated.pdf

# More threads for large PDFs
python translate_pdf.py document.pdf -w 20

# OpenAI with 5 threads
python translate_pdf.py document.pdf -e openai -w 5
```

## How it works

1. **Extract** — Reads all text blocks with their position, font, size and color
2. **Translate** — Translates all blocks in parallel (multithreaded)
3. **Apply** — Redacts original text (white fill, preserving images) and inserts the translated text at the same position with matching style

If the French text is longer than the original, the font size is progressively reduced (down to 60% of the original) to fit within the same bounding box.
