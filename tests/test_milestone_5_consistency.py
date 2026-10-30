"""Comprehensive 4-Way Cross-Module Consistency Matrix & Integrity Audit (Milestone 5: F18~F22).

Verifies 100% agreement across:
1. Master Thesis (Paper/Main/contents/chapter04_experiments.tex, main.pdf, main.log)
2. Oral Defense Beamer (Paper/Beamer/defense.tex, defense.pdf, defense.log)
3. Figures (Paper/Main/figures/user_141540.pdf, user_58074.pdf, user_83279.pdf, demo_screenshot.png)
4. Demo System (demo/web/demo_cases.json, demo/server.py)
5. Ground-Truth Dataset (data/processed/stats.pkl)

Covers:
- F18: Thesis 4-pass XeLaTeX compilation (0 undefined citations, 97 pages)
- F19: Beamer 2-pass XeLaTeX compilation (0 fatal errors, 33 slides)
- F20: Demo Data Remap Verification (1,364 USER nodes, 0 Raw ID leaks)
- F21: Visual Anti-Overflow Verification (11 target pages, bounding boxes within canvas)
- F22: 4-Way Cross-Module Consistency for Case A, Case B, Case C, and Demo Case 0
"""

from pathlib import Path
import json
import pickle
import re
import fitz
from PIL import Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def stats_data():
    stats_path = PROJECT_ROOT / "data" / "processed" / "stats.pkl"
    assert stats_path.exists(), f"stats.pkl not found: {stats_path}"
    with open(stats_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def demo_cases_json():
    json_path = PROJECT_ROOT / "demo" / "web" / "demo_cases.json"
    assert json_path.exists(), f"demo_cases.json not found: {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def thesis_tex_text():
    tex_path = PROJECT_ROOT / "Paper" / "Main" / "contents" / "chapter04_experiments.tex"
    assert tex_path.exists(), f"Thesis chapter 4 not found: {tex_path}"
    with open(tex_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def beamer_tex_text():
    tex_path = PROJECT_ROOT / "Paper" / "Beamer" / "defense.tex"
    assert tex_path.exists(), f"Beamer defense.tex not found: {tex_path}"
    with open(tex_path, "r", encoding="utf-8") as f:
        return f.read()


# =========================================================================
# F18 & F19: Compilation & Log Audits
# =========================================================================

def test_f18_thesis_pdf_compilation_and_log():
    """F18: Verify Thesis PDF exists, 97 pages, 0 undefined citations/labels."""
    pdf_path = PROJECT_ROOT / "Paper" / "Main" / "main.pdf"
    log_path = PROJECT_ROOT / "Paper" / "Main" / "main.log"
    assert pdf_path.exists(), f"Thesis main.pdf not found: {pdf_path}"
    assert log_path.exists(), f"Thesis main.log not found: {log_path}"

    doc = fitz.open(pdf_path)
    # Milestone 5 initial draft was 97 pages; with full Appendix D and updated TOC it is 99 pages
    assert len(doc) in (97, 99), f"Expected 97 or 99 pages in main.pdf, got {len(doc)}"

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()

    # Verify no undefined citations
    undefined_citations = re.findall(r"Citation `(.*?)' on page \d+ undefined", log_content)
    assert len(undefined_citations) == 0, f"Found undefined citations: {undefined_citations}"

    # Verify no undefined references
    undefined_refs = re.findall(r"Reference `(.*?)' on page \d+ undefined", log_content)
    assert len(undefined_refs) == 0, f"Found undefined references: {undefined_refs}"

    # Verify labels did not change
    assert "Label(s) may have changed" not in log_content, "Labels changed warning in main.log"


def test_f19_beamer_pdf_compilation_and_log():
    """F19: Verify Beamer PDF exists, 33 slides, 0 fatal errors."""
    pdf_path = PROJECT_ROOT / "Paper" / "Beamer" / "defense.pdf"
    log_path = PROJECT_ROOT / "Paper" / "Beamer" / "defense.log"
    assert pdf_path.exists(), f"Beamer defense.pdf not found: {pdf_path}"
    assert log_path.exists(), f"Beamer defense.log not found: {log_path}"

    doc = fitz.open(pdf_path)
    assert len(doc) == 33, f"Expected 33 slides in defense.pdf, got {len(doc)}"

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()

    # Verify no TeX fatal errors (lines starting with '! ')
    fatal_errors = [line for line in log_content.splitlines() if line.startswith("! ")]
    assert len(fatal_errors) == 0, f"Found fatal errors in defense.log: {fatal_errors}"


# =========================================================================
# F20: Demo Data Remap & Server Verification
# =========================================================================

def test_f20_demo_data_remap_and_server(demo_cases_json):
    """F20: Verify zero Raw ID leaks in USER nodes across demo_cases.json and server.py."""
    user_cache = demo_cases_json.get("user_cache", {})
    assert len(user_cache) == 500, f"Expected 500 users in user_cache, got {len(user_cache)}"

    total_user_nodes = 0
    raw_leaks = []

    for uid_str, uentry in user_cache.items():
        uid = uentry["user_id"]
        for d_key, d_data in uentry.get("depths", {}).items():
            if not isinstance(d_data, dict):
                continue
            for exp in d_data.get("explanations", []):
                for node in exp.get("path_details", []):
                    if node.get("type") == "USER":
                        total_user_nodes += 1
                        remap_id = node.get("id_remapped")
                        orig_id = node.get("id_original")
                        name = node.get("name")

                        if name != f"User {remap_id}":
                            raw_leaks.append(f"User node name '{name}' != 'User {remap_id}'")
                        if orig_id != remap_id and f"User {orig_id}" in name:
                            raw_leaks.append(f"Raw ID leak: 'User {orig_id}' in name '{name}'")
                        if orig_id != remap_id and f"User {orig_id}" in exp.get("path_description", ""):
                            raw_leaks.append(f"Raw ID leak: 'User {orig_id}' in path_description")

    assert total_user_nodes == 1364, f"Expected exactly 1,364 USER nodes in user_cache, found {total_user_nodes}"
    assert len(raw_leaks) == 0, f"Found {len(raw_leaks)} raw ID leaks: {raw_leaks[:5]}"

    # Verify server.py online inference enforcement
    server_py = PROJECT_ROOT / "demo" / "server.py"
    with open(server_py, "r", encoding="utf-8") as f:
        server_code = f.read()

    assert 'if n_type == "USER":\n                            n_name = f"User {node_id}"' in server_code
    assert 'user_name = f"User {uid}"' in server_code


# =========================================================================
# F21: PyMuPDF Visual Inspection & Anti-Overflow Matrix
# =========================================================================

def test_f21_visual_anti_overflow():
    """F21: Verify bounding boxes are strictly within canvas for key pages."""
    thesis_pdf = PROJECT_ROOT / "Paper" / "Main" / "main.pdf"
    beamer_pdf = PROJECT_ROOT / "Paper" / "Beamer" / "defense.pdf"

    thesis_doc = fitz.open(thesis_pdf)
    beamer_doc = fitz.open(beamer_pdf)

    # 1. Thesis key pages
    thesis_pages = [32, 48, 49, 50, 69, 71]
    for p_num in thesis_pages:
        page = thesis_doc[p_num - 1]
        w, h = page.rect.width, page.rect.height
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, _, _ = b
            assert x0 >= -1.0 and y0 >= -1.0 and x1 <= w + 1.0 and y1 <= h + 1.0, (
                f"Thesis page {p_num} block exceeds canvas: {b[:4]} on canvas ({w}, {h})"
            )

    # 2. Beamer key slides
    beamer_slides = [19, 20, 25, 28, 32, 33]
    for s_num in beamer_slides:
        page = beamer_doc[s_num - 1]
        w, h = page.rect.width, page.rect.height
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, _, _ = b
            assert x0 >= -1.0 and y0 >= -1.0 and x1 <= w + 1.0 and y1 <= h + 1.0, (
                f"Beamer slide {s_num} block exceeds canvas: {b[:4]} on canvas ({w}, {h})"
            )


# =========================================================================
# F22: 4-Way Cross-Module Consistency Matrix (Case A, B, C, Demo Case 0)
# =========================================================================

def test_case_a_4way_consistency(thesis_tex_text, beamer_tex_text, stats_data):
    """Case A (Direct Link): 4-Way consistency across Thesis, Beamer, Figure, and stats.pkl."""
    user_map = stats_data["user_map"]
    item_map = stats_data["item_map"]

    # 1. Dataset mapping
    assert user_map.classes_[141540] == 2410864
    assert item_map.classes_[224644] == 486496

    # 2. Thesis text
    assert "2410864" in thesis_tex_text
    assert "bacon lattice tomato muffins" in thesis_tex_text
    assert "0.952" in thesis_tex_text
    assert "0.767" in thesis_tex_text
    assert "0.2415" in thesis_tex_text
    assert "0.0004" in thesis_tex_text
    assert "figures/user_141540.pdf" in thesis_tex_text

    # 3. Beamer text (Appendix D1)
    assert "User \\#2410864" in beamer_tex_text
    assert "\\texttt{bacon lattice tomato muffins}" in beamer_tex_text
    assert "預測機率 0.952" in beamer_tex_text
    assert "\\alpha = 0.767" in beamer_tex_text
    assert "F^+ = 0.2415" in beamer_tex_text
    assert "F^- = 0.0004" in beamer_tex_text
    assert "figures/user_141540.pdf" in beamer_tex_text

    # 4. Figure content
    fig_path = PROJECT_ROOT / "Paper" / "Main" / "figures" / "user_141540.pdf"
    assert fig_path.exists()
    doc = fitz.open(fig_path)
    fig_text = doc[0].get_text()
    assert "0.7670" in fig_text
    assert "0.7538" not in fig_text  # Old obsolete weight must not exist
    assert "User 2410864" in fig_text
    assert "bacon lattice" in fig_text


def test_case_b_4way_consistency(thesis_tex_text, beamer_tex_text, stats_data):
    """Case B (Collaborative Filtering): 4-Way consistency across Thesis, Beamer, Figure, and stats.pkl."""
    user_map = stats_data["user_map"]
    item_map = stats_data["item_map"]

    # 1. Dataset mapping
    assert user_map.classes_[58074] == 573230
    assert item_map.classes_[63032] == 108383

    # 2. Thesis text
    assert "573230" in thesis_tex_text
    assert "joanne's creamy bruschetta" in thesis_tex_text
    assert "creamy cajun chicken pasta" in thesis_tex_text
    assert "User 603504" in thesis_tex_text
    assert "0.0006" in thesis_tex_text
    assert "0.2061" in thesis_tex_text
    assert "0.0019" in thesis_tex_text
    assert "figures/user_58074.pdf" in thesis_tex_text

    # 3. Beamer text (Appendix D2)
    assert "User \\#573230" in beamer_tex_text
    assert "\\texttt{joanne's creamy bruschetta}" in beamer_tex_text
    assert "預測機率 0.952" in beamer_tex_text
    assert "F^+ = 0.2061" in beamer_tex_text
    assert "F^- = 0.0019" in beamer_tex_text
    assert "figures/user_58074.pdf" in beamer_tex_text

    # 4. Figure content
    fig_path = PROJECT_ROOT / "Paper" / "Main" / "figures" / "user_58074.pdf"
    assert fig_path.exists()
    doc = fitz.open(fig_path)
    fig_text = doc[0].get_text()
    assert "User 573230" in fig_text
    assert "creamy cajun" in fig_text
    assert "User 603504" in fig_text
    assert "joanne s" in fig_text
    assert "0.0006" in fig_text


def test_case_c_4way_consistency(thesis_tex_text, beamer_tex_text, stats_data):
    """Case C (KG Semantic Path): 4-Way consistency across Thesis, Beamer, Figure, and stats.pkl."""
    user_map = stats_data["user_map"]
    item_map = stats_data["item_map"]

    # 1. Dataset mapping
    assert user_map.classes_[83279] == 963993
    assert item_map.classes_[13816] == 26254

    # 2. Thesis text
    assert "963993" in thesis_tex_text
    assert "fantastic tropical thai chicken salad" in thesis_tex_text
    assert "0.933" in thesis_tex_text
    assert "-0.001" in thesis_tex_text
    assert "0.5415" in thesis_tex_text
    assert "chicken" in thesis_tex_text
    assert "figures/user_83279.pdf" in thesis_tex_text

    # 3. Beamer text (Slide 18)
    assert "User \\#963993" in beamer_tex_text
    assert "\\texttt{fantastic tropical thai chicken salad}" in beamer_tex_text
    assert "預測機率 0.933" in beamer_tex_text
    assert "\\texttt{chicken}" in beamer_tex_text
    assert "F^+ = -0.001" in beamer_tex_text
    assert "F^- = 0.5415" in beamer_tex_text
    assert "figures/user_83279.pdf" in beamer_tex_text

    # 4. Figure content
    fig_path = PROJECT_ROOT / "Paper" / "Main" / "figures" / "user_83279.pdf"
    assert fig_path.exists()
    doc = fitz.open(fig_path)
    fig_text = doc[0].get_text()
    assert "User 963993" in fig_text
    assert "chicken" in fig_text
    assert "0.0003" in fig_text


def test_demo_case_0_cross_consistency(demo_cases_json, beamer_tex_text, stats_data):
    """Demo Case 0 (Deep Propagation): Consistency across demo_cases.json, screenshot, Beamer, and stats.pkl."""
    user_map = stats_data["user_map"]
    item_map = stats_data["item_map"]

    # 1. Dataset mapping
    assert user_map.classes_[17049] == 144146
    assert item_map.classes_[14829] == 27922

    # 2. Demo cases JSON
    c0 = demo_cases_json["academic_cases"][0]
    assert c0["case_id"] == "deep_propagation"
    assert c0["user_id"] == 17049
    assert c0["target_item_name"] == "roast pork with herb crust"

    d3 = c0["data"]["3"]
    assert round(d3["prob"], 4) == 0.9514
    assert len(d3["explanations"]) == 1
    exp = d3["explanations"][0]
    assert exp["path_details"][0]["name"] == "User 17049"
    assert "144146" not in exp["path_description"]
    assert "User 17049" in exp["path_description"]
    assert "pork" in exp["path_description"]
    assert "roast pork with herb crust" in exp["path_description"]

    # 3. Screenshot figure
    screenshot_path = PROJECT_ROOT / "Paper" / "Main" / "figures" / "demo_screenshot.png"
    assert screenshot_path.exists()
    assert screenshot_path.stat().st_size > 30000
    im = Image.open(screenshot_path)
    assert im.size in [(1440, 900), (2537, 1286)] or (im.width >= 1440 and im.height >= 900)

    # 4. Beamer Slide 19
    assert "demo_screenshot.png" in beamer_tex_text
    assert "56.74 MB / 190.8 ms" in beamer_tex_text
    assert "0.067 ms" in beamer_tex_text
    assert "11,205 倍" in beamer_tex_text
    assert "20.5 萬倍" in beamer_tex_text
