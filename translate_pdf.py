#!/usr/bin/env python3
"""Translate an English PDF to French in-place, preserving layout, images and styling."""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF


CHUNK_SIZE = 4500  # Google Translate character limit
MIN_FONT_SCALE = 0.6  # Minimum font size ratio when text overflows
DEFAULT_WORKERS = 10  # Default number of parallel translation threads


# --- Translation engines ---

def translate_google(text: str) -> str:
    """Translate text using Google Translate (free, no API key)."""
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="en", target="fr")
    chunks = chunk_text(text)
    translated = []
    for chunk in chunks:
        result = translator.translate(chunk)
        translated.append(result if result is not None else chunk)
    return "\n".join(translated)


def translate_openai(text: str) -> str:
    """Translate text using OpenAI API."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional translator. Translate the following English text to French. Return only the translated text, nothing else.",
            },
            {"role": "user", "content": text},
        ],
    )
    result = response.choices[0].message.content
    return result if result is not None else text


ENGINES = {
    "google": translate_google,
    "openai": translate_openai,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate an English PDF to French (in-place layout)."
    )
    parser.add_argument("input", help="Input PDF file path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output PDF file path (default: <input>_fr.pdf)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel translation threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=ENGINES.keys(),
        default="google",
        help="Translation engine (default: google)",
    )
    return parser.parse_args()


def chunk_text(text: str, max_len: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks of at most max_len characters, breaking at newlines."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(line) > max_len:
            if current:
                chunks.append(current)
                current = ""
            words = line.split(" ")
            part = ""
            for word in words:
                candidate = f"{part} {word}" if part else word
                if len(candidate) > max_len:
                    if part:
                        chunks.append(part)
                    part = word
                else:
                    part = candidate
            if part:
                current = part
            continue

        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_len:
            chunks.append(current)
            current = line
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def translate_text(text: str, engine_fn) -> str:
    """Translate text using the given engine function."""
    if not text.strip():
        return text
    return engine_fn(text)


def map_font(font_name: str, flags: int) -> str:
    """Map a PDF font name to a PyMuPDF built-in font.

    Built-in fonts: helv, hebo, heit, hebi, tiro, tibo, tiit, tibi, cour, cobo, coit, cobi
    """
    is_bold = bool(flags & 2 ** 4)  # bit 4 = bold
    is_italic = bool(flags & 2 ** 1)  # bit 1 = italic

    name_lower = font_name.lower() if font_name else ""

    if "courier" in name_lower or "mono" in name_lower:
        if is_bold and is_italic:
            return "cobi"
        if is_bold:
            return "cobo"
        if is_italic:
            return "coit"
        return "cour"

    if "times" in name_lower or "serif" in name_lower or "georgia" in name_lower:
        if is_bold and is_italic:
            return "tibi"
        if is_bold:
            return "tibo"
        if is_italic:
            return "tiit"
        return "tiro"

    # Default to Helvetica family
    if is_bold and is_italic:
        return "hebi"
    if is_bold:
        return "hebo"
    if is_italic:
        return "heit"
    return "helv"


def int_to_color(color_int: int) -> tuple:
    """Convert an integer color value to an (r, g, b) tuple with values 0-1."""
    r = ((color_int >> 16) & 0xFF) / 255.0
    g = ((color_int >> 8) & 0xFF) / 255.0
    b = (color_int & 0xFF) / 255.0
    return (r, g, b)


def extract_blocks(page: fitz.Page) -> list[dict]:
    """Extract text blocks with their metadata from a page."""
    blocks = []
    page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    for block in page_dict["blocks"]:
        if block["type"] != 0:  # skip non-text blocks (images, etc.)
            continue

        # Collect all text and dominant style from the block
        full_text = ""
        dominant_font = "helv"
        dominant_size = 11.0
        dominant_color = (0, 0, 0)
        dominant_flags = 0
        max_chars = 0

        for line in block["lines"]:
            line_text = ""
            for span in line["spans"]:
                span_text = span["text"]
                line_text += span_text
                # Track the most common style (by character count)
                if len(span_text) > max_chars:
                    max_chars = len(span_text)
                    dominant_font = span["font"]
                    dominant_size = span["size"]
                    dominant_color = int_to_color(span["color"])
                    dominant_flags = span["flags"]
            full_text += line_text + "\n"

        full_text = full_text.rstrip("\n")

        if not full_text.strip():
            continue

        blocks.append({
            "bbox": fitz.Rect(block["bbox"]),
            "text": full_text,
            "font": dominant_font,
            "size": dominant_size,
            "color": dominant_color,
            "flags": dominant_flags,
        })

    return blocks


def apply_translations(page: fitz.Page, translated_blocks: list[dict]):
    """Redact original text and insert translations on a page."""
    # Redact original text (white fill to erase it)
    for block in translated_blocks:
        page.add_redact_annot(block["bbox"], fill=(1, 1, 1))

    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    # Insert translated text
    for block in translated_blocks:
        bbox = block["bbox"]
        mapped_font = map_font(block["font"], block["flags"])
        font_size = block["size"]
        color = block["color"]
        translated = block["translated"]

        # Try inserting; if it overflows, reduce font size progressively
        rc = -1
        current_size = font_size
        min_size = font_size * MIN_FONT_SCALE

        while current_size >= min_size:
            rc = page.insert_textbox(
                bbox,
                translated,
                fontname=mapped_font,
                fontsize=current_size,
                color=color,
                align=fitz.TEXT_ALIGN_LEFT,
            )
            if rc >= 0:  # rc >= 0 means text fitted
                break
            current_size -= 0.5

        # If still doesn't fit, insert anyway at minimum size
        if rc < 0:
            page.insert_textbox(
                bbox,
                translated,
                fontname=mapped_font,
                fontsize=min_size,
                color=color,
                align=fitz.TEXT_ALIGN_LEFT,
            )


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.is_file():
        print(f"Error: file not found — {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix.lower() != ".pdf":
        print(f"Error: not a PDF file — {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(
        input_path.with_stem(input_path.stem + "_fr")
    )

    engine_name = args.engine
    engine_fn = ENGINES[engine_name]
    workers = args.workers

    if engine_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required for openai engine.", file=sys.stderr)
        sys.exit(1)

    print(f"Opening {input_path}...")
    doc = fitz.open(str(input_path))
    total = len(doc)

    if total == 0:
        print("Error: PDF has no pages.", file=sys.stderr)
        sys.exit(1)

    # --- Phase 1: Extract all blocks from all pages (fast, sequential) ---
    print(f"Extracting text from {total} page(s)...")
    page_blocks = {}  # page_index -> list of blocks
    for i, page in enumerate(doc):
        blocks = extract_blocks(page)
        if blocks:
            page_blocks[i] = blocks

    total_blocks = sum(len(b) for b in page_blocks.values())
    if total_blocks == 0:
        print("Error: no extractable text found in the PDF.", file=sys.stderr)
        sys.exit(1)

    # --- Phase 2: Translate all blocks in parallel (slow, I/O-bound) ---
    print(f"Translating {total_blocks} text blocks with {workers} threads ({engine_name})...")

    # Build a flat list of (page_index, block_index, text) for the executor
    tasks = []
    for page_idx, blocks in page_blocks.items():
        for block_idx, block in enumerate(blocks):
            tasks.append((page_idx, block_idx, block["text"]))

    translated_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_key = {}
        for page_idx, block_idx, text in tasks:
            future = executor.submit(translate_text, text, engine_fn)
            future_to_key[future] = (page_idx, block_idx)

        for future in as_completed(future_to_key):
            page_idx, block_idx = future_to_key[future]
            page_blocks[page_idx][block_idx]["translated"] = future.result()
            translated_count += 1
            print(f"\r  Translated {translated_count}/{total_blocks} blocks", end="", flush=True)

    print()  # newline after progress

    # --- Phase 3: Apply redactions and insert translations (fast, sequential) ---
    print("Applying translations to PDF...")
    sorted_pages = sorted(page_blocks.keys())
    for count, page_idx in enumerate(sorted_pages, 1):
        page = doc[page_idx]
        apply_translations(page, page_blocks[page_idx])
        pct = count * 100 // len(sorted_pages)
        bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
        print(f"\r  {bar} {pct:3d}% ({count}/{len(sorted_pages)} pages)", end="", flush=True)
    print()

    print(f"Saving translated PDF...")
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    print(f"Done! Output: {output_path}")


if __name__ == "__main__":
    main()
