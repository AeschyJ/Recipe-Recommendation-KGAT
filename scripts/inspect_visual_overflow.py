"""Automated visual inspection and anti-overflow validation using PyMuPDF (fitz).

Validates:
1. Thesis:
   - §2.4 / §2.5 Golden Triangle literature (page 32)
   - §3.6 Algorithm 1 pseudocode (page 48)
   - §3.6 Theorem 3.1 (page 49)
   - §4.6 Table 4.12 (page 69)
   - §4.6 Table 4.13 (page 71)
2. Beamer:
   - Slide 19: CSR box + demo_screenshot.png
   - Slide 20: 2x2 contribution cards (height=2.35cm)
   - Slide 25: Appendix B1 (61 ingredients)
   - Slide 28: Appendix D1 (alpha = 0.767)
   - Slide 32: Appendix G1 (18-case ablation table)
   - Slide 33: Appendix G2 (equivalence proof & L1 cache)

Renders all pages to 150 DPI PNGs in output/visual_inspection/ and checks bounding boxes.
"""

from pathlib import Path
import re
import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "visual_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_and_render_page(doc, page_num_1indexed, output_filename, label, dpi=150):
    page = doc[page_num_1indexed - 1]
    rect = page.rect
    width, height = rect.width, rect.height

    # Render to PNG
    pix = page.get_pixmap(dpi=dpi)
    out_path = OUTPUT_DIR / output_filename
    pix.save(str(out_path))

    # Inspect all text blocks
    blocks = page.get_text("blocks")
    overflows = []
    min_x, min_y, max_x, max_y = width, height, 0, 0

    for b in blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        min_x = min(min_x, x0)
        min_y = min(min_y, y0)
        max_x = max(max_x, x1)
        max_y = max(max_y, y1)

        # Check if text extends outside page boundary
        # Allowing a 1-point tolerance for floating-point rounding
        if x0 < -1.0 or y0 < -1.0 or x1 > width + 1.0 or y1 > height + 1.0:
            overflows.append({
                "block": (x0, y0, x1, y1),
                "text_snippet": text.strip()[:60].replace("\n", " "),
                "reason": "Exceeds page boundary",
            })

    result = {
        "label": label,
        "page_num": page_num_1indexed,
        "output_file": str(out_path),
        "canvas_size": (width, height),
        "content_bbox": (min_x, min_y, max_x, max_y),
        "total_blocks": len(blocks),
        "overflow_count": len(overflows),
        "overflows": overflows,
        "status": "PASS" if len(overflows) == 0 else "FAIL",
    }
    return result


def run_all_inspections():
    thesis_pdf = PROJECT_ROOT / "Paper" / "Main" / "main.pdf"
    beamer_pdf = PROJECT_ROOT / "Paper" / "Beamer" / "defense.pdf"

    assert thesis_pdf.exists(), f"Thesis PDF not found: {thesis_pdf}"
    assert beamer_pdf.exists(), f"Beamer PDF not found: {beamer_pdf}"

    thesis_doc = fitz.open(thesis_pdf)
    beamer_doc = fitz.open(beamer_pdf)

    thesis_targets = [
        (32, "thesis_p32_sec2_golden_triangle.png", "Thesis §2.4/§2.5 Golden Triangle"),
        (48, "thesis_p48_sec3_algorithm1_intro.png", "Thesis §3.6 Algorithm 1 text intro"),
        (49, "thesis_p49_sec3_theorem3_1.png", "Thesis §3.6 Theorem 3.1 & Proof"),
        (50, "thesis_p50_sec3_algorithm1_float.png", "Thesis §3.6 Algorithm 1 pseudocode"),
        (69, "thesis_p69_sec4_table4_12.png", "Thesis §4.6 Table 4.12 Construction Overhead"),
        (71, "thesis_p71_sec4_table4_13.png", "Thesis §4.6 Table 4.13 18-Group Ablation"),
    ]

    beamer_targets = [
        (19, "beamer_slide19_csr_demo.png", "Beamer Slide 19: CSR Box & Demo Screenshot"),
        (20, "beamer_slide20_contributions_2x2.png", "Beamer Slide 20: 2x2 Core Contributions"),
        (25, "beamer_slide25_appendix_b1_ingredients.png", "Beamer Slide 25: Appendix B1 61 Ingredients"),
        (28, "beamer_slide28_appendix_d1_case_a.png", "Beamer Slide 28: Appendix D1 Case A (alpha=0.767)"),
        (32, "beamer_slide32_appendix_g1_ablation.png", "Beamer Slide 32: Appendix G1 18-Group Ablation"),
        (33, "beamer_slide33_appendix_g2_equivalence.png", "Beamer Slide 33: Appendix G2 Equivalence Proof & L1 Cache"),
    ]

    print("=== Visual Inspection & Anti-Overflow Evaluation ===")
    results = []

    print("\n--- 1. Thesis Key Pages ---")
    for p_num, filename, label in thesis_targets:
        res = inspect_and_render_page(thesis_doc, p_num, filename, label)
        results.append(res)
        w, h = res["canvas_size"]
        x0, y0, x1, y1 = res["content_bbox"]
        print(f"[{res['status']}] {label} (p.{p_num}): canvas=({w:.1f}x{h:.1f}), bbox=({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}), overflows={res['overflow_count']}")

    print("\n--- 2. Beamer Key Slides ---")
    for s_num, filename, label in beamer_targets:
        res = inspect_and_render_page(beamer_doc, s_num, filename, label)
        results.append(res)
        w, h = res["canvas_size"]
        x0, y0, x1, y1 = res["content_bbox"]
        print(f"[{res['status']}] {label} (slide {s_num}): canvas=({w:.1f}x{h:.1f}), bbox=({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}), overflows={res['overflow_count']}")

    all_passed = all(r["status"] == "PASS" for r in results)
    print(f"\nFinal Verdict: {'ALL PASS (Zero Overflow Detected)' if all_passed else 'FAIL'}")
    return all_passed, results


if __name__ == "__main__":
    success, _ = run_all_inspections()
    if not success:
        exit(1)
