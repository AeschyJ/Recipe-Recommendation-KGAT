"""Unit and Forensic Verification Tests for Milestone 4 (R4: F15, F16, F17).

Verifies:
1. Paper/Beamer/defense.tex syntax, structure, terminology cleaning, 61 ingredients,
   Slide 19 insertion, Slide 20 2x2 grid (height=2.35cm), Appendix G1 & G2.
2. Paper/Beamer/defense.pdf compilation integrity, 33 total slides.
3. Specific slide contents in the compiled PDF via PyMuPDF (fitz).
"""

from pathlib import Path
import re
import fitz
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_beamer_pdf_compilation_and_slide_count():
    pdf_path = PROJECT_ROOT / "Paper" / "Beamer" / "defense.pdf"
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    assert pdf_path.stat().st_size > 100000, f"PDF file size too small: {pdf_path.stat().st_size}"

    doc = fitz.open(pdf_path)
    # Slide count must increase from 30 to 33
    assert len(doc) == 33, f"Expected 33 slides in defense.pdf, got {len(doc)}"

def test_f15_errata_and_terminology_sweep():
    tex_path = PROJECT_ROOT / "Paper" / "Beamer" / "defense.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Appendix D1 attention weight verification
    d1_match = re.search(r"附錄 D1: Case A.*?\\item \\textbf\{注意力權重\}：(.*?)\n", text, re.DOTALL)
    assert d1_match is not None, "Appendix D1 attention weight not found"
    assert "\\alpha = 0.767" in d1_match.group(1), f"Appendix D1 attention weight is not \\alpha = 0.767: {d1_match.group(1)}"

    # 2. Appendix B1 61 ingredients verification
    b1_match = re.search(r"高頻食材過濾清單.*?\\texttt\{(.*?)\}", text, re.DOTALL)
    assert b1_match is not None, "Appendix B1 ingredients not found"
    ingredients = [i.strip() for i in b1_match.group(1).split(",")]
    assert len(ingredients) == 61, f"Expected 61 ingredients in Appendix B1, got {len(ingredients)}"

    # 7 restored missing ingredients
    missing = ["onion", "cinnamon", "parsley", "garlic clove", "carrots", "garlic cloves", "green onions"]
    for m in missing:
        assert m in ingredients, f"Restored ingredient '{m}' missing in Appendix B1 list"

    # 3. Terminology cleaning checks
    # Target terms must be present
    assert "殘缺拓樸" in text, "Term '殘缺拓樸' missing"
    assert "隱含正規化" in text, "Term '隱含正規化' missing"
    assert "附錄 B1: 食材 (Ingredients) 雜訊過濾清單" in text, "Term '附錄 B1: 食材 (Ingredients) 雜訊過濾清單' missing"
    assert "基礎調味料雜訊清單" in text, "Term '基礎調味料雜訊清單' missing"
    assert "附錄 B2: 標籤 (Tags) 雜訊過濾清單" in text, "Term '附錄 B2: 標籤 (Tags) 雜訊過濾清單' missing"

    # Forbidden uncleaned terms must NOT exist anywhere in defense.tex
    assert "拓撲" not in text, "Forbidden term '拓撲' still exists in defense.tex"
    assert "隱含正則化" not in text, "Forbidden term '隱含正則化' still exists in defense.tex"
    assert "噪音過濾清單" not in text, "Forbidden term '噪音過濾清單' still exists in defense.tex"
    assert "噪音清單" not in text, "Forbidden term '噪音清單' still exists in defense.tex"

def test_f16_slide_19_and_slide_20_structure():
    tex_path = PROJECT_ROOT / "Paper" / "Beamer" / "defense.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Slide 19 checks
    assert "RQ3 延伸實踐：大規模 CKG 即時路徑萃取加速與互動展示" in text
    assert "CSR 雙向局部交集向量化演算法效益" in text
    assert "demo_screenshot.png" in text
    assert "56.74 MB / 190.8 ms" in text
    assert "0.067 ms" in text
    assert "11,205 倍" in text
    assert "20.5 萬倍" in text
    assert "100\\% PASS" in text
    assert "0.145 ms" in text

    # Slide 20 checks
    assert "結論與核心研究貢獻" in text
    assert "貢獻一：傳播深度實證與邊際增益" in text
    assert "貢獻二：淺層 KG 資訊污染現象" in text
    assert "貢獻三：解釋路徑忠實度量化評估" in text
    assert "貢獻四：即時路徑萃取與互動展示" in text

    # Critical requirement: height=2.35cm on all 4 cards to prevent Beamer vertical overflow
    card_heights = re.findall(r"\\begin\{mybox\}\[[^\]]*height=2\.35cm[^\]]*\]\{ntublue\}\{\{貢獻[一二三四]", text)
    assert len(card_heights) == 4, f"Expected 4 cards with height=2.35cm on Slide 20, found {len(card_heights)}"

def test_f17_appendix_g1_and_g2_structure():
    tex_path = PROJECT_ROOT / "Paper" / "Beamer" / "defense.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Appendix G1 checks
    assert "附錄 G1: 大規模 CKG 路徑萃取基準測試完整數據" in text
    assert "18 組代表性案例 $2 \\times 2$ 因子消融延遲表 (Traversal Latency Benchmark)" in text
    assert "深度傳播成功案例 (User 17049)" in text
    assert "高必要性案例 (User 101612)" in text
    assert "超熱門食材實體 (2組平均, 度數 22,095)" in text
    assert "11,205.2$\\times$" in text or "11,205.2" in text

    # Appendix G2 checks
    assert "附錄 G2: CSR 雙向局部交集演算法等價性證明與快取親和性" in text
    assert "三部圖定界與雙射等價定理" in text or "三部圖定界與路徑等價性質" in text
    assert "硬體快取親和性與超節點免疫" in text
    assert "三部圖排他性" in text
    assert "雙向相交雙射性" in text
    assert "非對稱 L1 Cache 屏障" in text
    assert "旁路超級節點展開" in text

def test_pdf_rendered_slide_content():
    pdf_path = PROJECT_ROOT / "Paper" / "Beamer" / "defense.pdf"
    doc = fitz.open(pdf_path)

    # Slide 19 (page 19, 0-indexed: 18)
    slide_19_text = re.sub(r"\s+", "", doc[18].get_text())
    assert "RQ3延伸實踐：大規模CKG即時路徑萃取加速與互動展示" in slide_19_text
    assert "CSR雙向局部交集向量化演算法" in slide_19_text

    # Slide 20 (page 20, 0-indexed: 19)
    slide_20_text = re.sub(r"\s+", "", doc[19].get_text())
    assert "結論與核心研究貢獻" in slide_20_text
    assert "貢獻一" in slide_20_text
    assert "貢獻二" in slide_20_text
    assert "貢獻三" in slide_20_text
    assert "貢獻四" in slide_20_text

    # Appendix G1 (page 32, 0-indexed: 31)
    slide_32_text = re.sub(r"\s+", "", doc[31].get_text())
    assert "附錄G1" in slide_32_text
    assert "11,205.2" in slide_32_text

    # Appendix G2 (page 33, 0-indexed: 32)
    slide_33_text = re.sub(r"\s+", "", doc[32].get_text())
    assert "附錄G2" in slide_33_text
    assert "三部圖定界與雙射等價定理" in slide_33_text or "三部圖定界與路徑等價性質" in slide_33_text
