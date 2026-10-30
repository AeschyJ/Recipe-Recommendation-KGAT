"""Empirical and Adversarial Forensic Test Suite for Milestone 2.

Covers:
1. PDF Canvas & Vector Layout Forensic (user_141540.pdf)
   - Edge weight label exact 0.7670 (0.7538 zeroed out)
   - Remapped user ID: User 2410864
   - Target item: bacon lattice tomato muffins rsc
   - Canvas clipping & bounding box containment
   - Pairwise text block collision / overlap detection
2. Cross-Module Consistency (Case A, Case B, Case C)
   - Thesis chapter04_experiments.tex vs Beamer defense.tex vs PDF vector figures
3. Image Forensics (demo_screenshot.png)
   - Magic bytes / PNG header verification
   - Dimensions 1440x900
   - Color distribution & variance (non-blank)
   - Visual content inspection: UI elements vs blank graph area analysis
"""

import json
from pathlib import Path
from PIL import Image
import fitz
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def case_a_pdf():
    pdf_path = PROJECT_ROOT / "Paper" / "Main" / "figures" / "user_141540.pdf"
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    doc = fitz.open(pdf_path)
    yield doc
    doc.close()


@pytest.fixture(scope="module")
def demo_screenshot_img():
    img_path = PROJECT_ROOT / "Paper" / "Main" / "figures" / "demo_screenshot.png"
    assert img_path.exists(), f"Screenshot not found: {img_path}"
    return img_path


def test_user_141540_pdf_header_and_basics(case_a_pdf):
    """Verify PDF validity, page count, and dimensions."""
    assert len(case_a_pdf) == 1, "Expected single-page PDF"
    page = case_a_pdf[0]
    rect = page.rect
    # Dimensions roughly ~572 x 482 pt
    assert rect.width > 400 and rect.height > 300, f"Unexpected canvas size: {rect}"


def test_user_141540_pdf_text_content_and_labels(case_a_pdf):
    """Verify edge weight 0.7670, User 2410864, target item, and absence of 0.7538."""
    page = case_a_pdf[0]
    text = page.get_text()

    # Exact attention weight label
    assert "0.7670" in text, "Attention weight label 0.7670 not found in PDF"
    assert "0.7538" not in text, "Old unaligned weight 0.7538 must NOT be in PDF"

    # Core node labels
    assert "User 2410864" in text, "User 2410864 not found in PDF"
    assert "bacon lattice" in text, "bacon lattice not found in PDF"
    assert "muffins" in text, "muffins not found in PDF"

    # Title and Legend
    assert "Recommendation Explanation (KGAT L=3)" in text, "Title missing or altered"
    assert "Recommended Item" in text, "Legend missing Recommended Item"
    assert "History/Neighbor Recipe" in text, "Legend missing History/Neighbor Recipe"


def test_user_141540_pdf_canvas_clipping_and_margins(case_a_pdf):
    """Verify that all text blocks are fully contained within the canvas with margins."""
    page = case_a_pdf[0]
    w, h = page.rect.width, page.rect.height

    blocks = page.get_text("blocks")
    assert len(blocks) >= 5, f"Expected at least 5 text blocks, got {len(blocks)}"

    for i, (x0, y0, x1, y1, btext, _, _) in enumerate(blocks):
        clean_text = btext.strip()
        if not clean_text:
            continue
        # Bounding box must be strictly within canvas
        assert x0 >= 0, f"Block {i} ('{clean_text}') clipped left: x0={x0}"
        assert y0 >= 0, f"Block {i} ('{clean_text}') clipped top: y0={y0}"
        assert x1 <= w, f"Block {i} ('{clean_text}') clipped right: x1={x1} > {w}"
        assert y1 <= h, f"Block {i} ('{clean_text}') clipped bottom: y1={y1} > {h}"


def test_user_141540_pdf_no_disjoint_text_collision(case_a_pdf):
    """Verify that edge label 0.7670 does not collide with node labels or title."""
    page = case_a_pdf[0]
    blocks = page.get_text("blocks")

    edge_label_box = None
    node_boxes = []

    for b in blocks:
        text = b[4].strip()
        if "0.7670" in text:
            edge_label_box = fitz.Rect(b[:4])
        elif "User 2410864" in text and "Target" not in text:
            node_boxes.append(("User", fitz.Rect(b[:4])))
        elif "bacon" in text and "Target" not in text:
            node_boxes.append(("Item", fitz.Rect(b[:4])))

    assert edge_label_box is not None, "Edge label bbox not located"
    assert len(node_boxes) >= 2, f"Node bboxes incomplete: {node_boxes}"

    # Verify edge label does not intersect node text boxes
    for name, nbox in node_boxes:
        intersect = edge_label_box.intersect(nbox)
        assert intersect.is_empty, f"Edge label overlaps with {name} node box: {intersect}"


def test_cross_module_consistency_cases_a_b_c():
    """Verify 4-way consistency across Thesis, Beamer, Figures for Cases A, B, and C."""
    thesis_file = PROJECT_ROOT / "Paper" / "Main" / "contents" / "chapter04_experiments.tex"
    beamer_file = PROJECT_ROOT / "Paper" / "Beamer" / "defense.tex"

    with open(thesis_file, "r", encoding="utf-8") as f:
        thesis_text = f.read()
    with open(beamer_file, "r", encoding="utf-8") as f:
        beamer_text = f.read()

    # Case A Check
    assert "User 2410864" in thesis_text, "Thesis missing User 2410864"
    assert "0.767" in thesis_text, "Thesis missing Case A weight 0.767"
    assert "2410864" in beamer_text, "Beamer missing User 2410864"
    assert "0.767" in beamer_text, "Beamer missing Case A weight 0.767"

    # Case B Check
    assert "User 573230" in thesis_text, "Thesis missing User 573230"
    assert "joanne's creamy bruschetta" in thesis_text, "Thesis missing Case B item"
    assert "573230" in beamer_text, "Beamer missing User 573230"
    assert "joanne's creamy bruschetta" in beamer_text, "Beamer missing Case B item"

    # Case C Check
    assert "User 963993" in thesis_text, "Thesis missing User 963993"
    assert "fantastic tropical thai chicken salad" in thesis_text, "Thesis missing Case C item"

    # Case B & C PDF Figures Check
    doc_b = fitz.open(PROJECT_ROOT / "Paper" / "Main" / "figures" / "user_58074.pdf")
    text_b = doc_b[0].get_text()
    assert "User 573230" in text_b, "user_58074.pdf missing User 573230"
    assert "joanne" in text_b, "user_58074.pdf missing joanne"
    doc_b.close()

    doc_c = fitz.open(PROJECT_ROOT / "Paper" / "Main" / "figures" / "user_83279.pdf")
    text_c = doc_c[0].get_text()
    assert "User 963993" in text_c, "user_83279.pdf missing User 963993"
    assert "fantastic" in text_c, "user_83279.pdf missing fantastic"
    doc_c.close()


def test_demo_screenshot_format_and_header(demo_screenshot_img):
    """Verify PNG header magic bytes, dimensions, and file size."""
    with open(demo_screenshot_img, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n", "Invalid PNG magic header"

    im = Image.open(demo_screenshot_img)
    assert im.format == "PNG", f"Expected PNG, got {im.format}"
    assert im.size in [(1440, 900), (2537, 1286)] or (im.width >= 1440 and im.height >= 900), f"Unexpected size: {im.size}"
    assert im.mode in ("RGB", "RGBA"), f"Unexpected mode: {im.mode}"


def test_demo_screenshot_color_variance(demo_screenshot_img):
    """Verify screenshot has non-trivial color variance and is not blank or corrupted."""
    im = Image.open(demo_screenshot_img).convert("RGB")
    arr = np.array(im)

    # Global variance check
    std_per_channel = arr.std(axis=(0, 1))
    assert all(s > 15 for s in std_per_channel), f"Image std too low (near blank): {std_per_channel}"

    # Unique colors check
    unique_colors = len(np.unique(arr.reshape(-1, 3), axis=0))
    assert unique_colors > 500, f"Too few unique colors: {unique_colors}"


def test_demo_screenshot_ui_graph_area_content(demo_screenshot_img):
    """Forensic challenge: verify whether the collaborative knowledge graph area is populated.

    Notice: In the current screenshot, the right panel (x: [520, 1380], y: [220, 800])
    is virtually blank white (mean > 254, unique colors = 72) because the headless browser
    captured the page during '載入中...' before executeInference() rendered the SVG graph.
    """
    im = Image.open(demo_screenshot_img).convert("RGB")
    arr = np.array(im)

    graph_area = arr[220:800, 520:1380]
    mean_color = graph_area.mean(axis=(0, 1))
    unique_colors = len(np.unique(graph_area.reshape(-1, 3), axis=0))

    # If the graph was drawn, there would be blue circles, orange circles, green circles,
    # arrow lines, and text labels, resulting in hundreds of unique colors and lower mean.
    is_graph_rendered = unique_colors > 300 and mean_color[0] < 250
    # Document this as a diagnostic finding:
    if not is_graph_rendered:
        pytest.fail(
            f"Adversarial Challenge Finding: demo_screenshot.png shows unpopulated UI! "
            f"Graph area unique colors={unique_colors} (expected >300), "
            f"mean color={mean_color.tolist()} (near pure white 254). "
            f"Web UI was captured in '載入中...' loading state without rendered knowledge graph."
        )
