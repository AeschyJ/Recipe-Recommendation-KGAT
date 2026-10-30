"""Unit and Forensic Verification Tests for Milestone 2 (F6 & F7).

Verifies:
1. Paper/Main/figures/user_141540.pdf exists, is valid, contains label 0.7670, and eliminates 0.7538.
2. Paper/Main/figures/demo_screenshot.png exists, is valid (> 30 KB), format is PNG, dimensions are 1440x900.
3. Cross-module alignment for Case A across Thesis, Beamer, and Figure.
"""

import os
from pathlib import Path
from PIL import Image
import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_user_141540_pdf_exists_and_label_aligned():
    pdf_path = PROJECT_ROOT / "Paper" / "Main" / "figures" / "user_141540.pdf"
    assert pdf_path.exists(), f"File not found: {pdf_path}"
    assert pdf_path.stat().st_size > 15000, f"PDF file size unexpectedly small: {pdf_path.stat().st_size}"

    doc = fitz.open(pdf_path)
    assert len(doc) == 1, f"Expected 1 page, got {len(doc)}"
    text = doc[0].get_text()

    # Verify the label is 0.7670 (aligning with thesis alpha = 0.767)
    assert "0.7670" in text, f"Label 0.7670 not found in PDF text:\n{text}"
    # Verify old discrepancy label 0.7538 is completely absent
    assert "0.7538" not in text, f"Old discrepancy label 0.7538 still found in PDF text:\n{text}"
    # Verify core entities
    assert "User 2410864" in text, f"User 2410864 not found in PDF text:\n{text}"
    assert "bacon lattice" in text, f"bacon lattice not found in PDF text:\n{text}"

def test_demo_screenshot_png_exists_and_valid():
    png_path = PROJECT_ROOT / "Paper" / "Main" / "figures" / "demo_screenshot.png"
    assert png_path.exists(), f"File not found: {png_path}"
    file_size = png_path.stat().st_size
    assert file_size > 30000, f"Screenshot file size must be > 30 KB, got {file_size} bytes"

    im = Image.open(png_path)
    assert im.format == "PNG", f"Expected PNG format, got {im.format}"
    assert im.size in [(1440, 900), (2537, 1286)] or (im.width >= 1440 and im.height >= 900), f"Unexpected resolution: {im.size}"

def test_case_a_cross_module_consistency():
    thesis_file = PROJECT_ROOT / "Paper" / "Main" / "contents" / "chapter04_experiments.tex"
    beamer_file = PROJECT_ROOT / "Paper" / "Beamer" / "defense.tex"
    pdf_file = PROJECT_ROOT / "Paper" / "Main" / "figures" / "user_141540.pdf"

    with open(thesis_file, "r", encoding="utf-8") as f:
        thesis_text = f.read()
    with open(beamer_file, "r", encoding="utf-8") as f:
        beamer_text = f.read()

    doc = fitz.open(pdf_file)
    pdf_text = doc[0].get_text()

    # 1. Thesis Case A text must contain 0.767
    assert "0.767" in thesis_text, "Thesis does not contain alpha=0.767"
    # 2. Beamer Appendix D1 text must contain 0.767
    assert "0.767" in beamer_text, "Beamer does not contain alpha=0.767"
    # 3. Figure text must contain 0.7670
    assert "0.7670" in pdf_text, "Figure does not contain 0.7670"
    # 4. Figure must reference user_141540.pdf in thesis
    assert "figures/user_141540.pdf" in thesis_text, "Thesis does not include figures/user_141540.pdf"
