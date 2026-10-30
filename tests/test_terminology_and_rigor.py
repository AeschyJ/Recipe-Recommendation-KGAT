"""Test Terminology, Rigor, and Literature Alignment for Thesis and Oral Defense.

Adheres strictly to ADR-008:
- Zero mainland Chinese terms or uncleaned phrases across all .tex files.
- Rigorous mathematical & Fidelity notation without flawed set difference notation.
- Baseline taxonomy and objective naming alignment.
- Verification of 2024-2026 literature survey in Section 2.4 defeating strawman critique.
- Verification of Appendix D formal proofs and Algorithm 1 pseudocode.
"""

from pathlib import Path
import re
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "Paper"
MAIN_TEX_DIR = PAPER_DIR / "Main"
BEAMER_TEX_DIR = PAPER_DIR / "Beamer"

FORBIDDEN_TERMS = [
    "網絡",
    "歸一化",
    "優化",
    "基線",
    "頂尖基準",
    "頂尖基線",
    "最強基線",
    "最強 Baseline",
    "最強的基準方法",
    "最強基準",
    "在線",
    "連線推論",
    "構建",
    "高並發",
    "指針",
    "正則化",
    "偽代碼",
    "噪音過濾清單",
    "噪音清單",
    "成交",
    "獨立運行",
    "拓撲",
    "交互",
    "內存",
    "數組",
    "緩存",
    "接口",
    "默認",
    "調用",
]


def test_zero_forbidden_terms_across_all_tex_files():
    """Verify 0 occurrences of forbidden mainland Chinese / uncleaned terms across all .tex files."""
    violations = []

    tex_files = list(PAPER_DIR.rglob("*.tex"))
    assert len(tex_files) > 0, "No .tex files found in Paper directory"

    for tex_file in tex_files:
        content = tex_file.read_text(encoding="utf-8")
        for term in FORBIDDEN_TERMS:
            if term in content:
                for line_idx, line in enumerate(content.splitlines(), 1):
                    if term in line:
                        rel_path = tex_file.relative_to(PROJECT_ROOT)
                        violations.append(f"{rel_path}:{line_idx} contains forbidden term '{term}': {line.strip()}")

    assert not violations, "Found forbidden terminology violations:\n" + "\n".join(violations)


def test_adr008_fidelity_notation_rigor():
    """Verify that defective Fidelity notations (p_{uv \\setminus S} or p_{uv \\mid S}) are completely absent."""
    tex_files = list(PAPER_DIR.rglob("*.tex"))
    defective_patterns = [
        r"p_\{uv\s*\\setminus\s*S\}",
        r"p_\{uv\s*\\mid\s*S\}",
        r"\\hat\{p\}_\{uv\s*\\setminus\s*S\}",
        r"\\hat\{p\}_\{uv\s*\\mid\s*S\}",
    ]

    violations = []
    for tex_file in tex_files:
        content = tex_file.read_text(encoding="utf-8")
        for pat in defective_patterns:
            matches = re.findall(pat, content)
            if matches:
                violations.append(f"{tex_file.relative_to(PROJECT_ROOT)} contains defective Fidelity notation matching '{pat}'")

    assert not violations, "Found defective Fidelity notation violations:\n" + "\n".join(violations)


def test_section_2_4_literature_survey_and_strawman_defense():
    """Verify Section 2.4 contains comprehensive 2024-2026 citations and refutes strawman comparison."""
    ch2_path = MAIN_TEX_DIR / "contents" / "chapter02_related_work.tex"
    assert ch2_path.exists(), "chapter02_related_work.tex does not exist"
    ch2_text = ch2_path.read_text(encoding="utf-8")

    assert "大規模圖可解釋性之計算瓶頸與即時推論挑戰" in ch2_text, "Section 2.4 heading missing"
    assert "稻草人比較" in ch2_text or "Strawman Comparison" in ch2_text, "Strawman comparison defense missing"

    # Verify key 2024-2026 papers and benchmarks are cited
    key_citations = [
        "liu2021dig",
        "lucic2022cfgnnexplainer",
        "hagberg2008networkx",
        "virtanen2020scipy",
        "li2025can",
        "rathee2024bagel",
        "kosan2024gnnxbench",
        "agarwal2023evaluating",
        "amara2024graphframex",
        "yang2024pr4sr",
        "liu2025cocounter",
        "ma2025kgexplainer",
        "sun2025expath",
        "ma2024xrec",
        "johnson2024xlightgcn",
        "zhang2024natural",
    ]
    for cite in key_citations:
        assert cite in ch2_text, f"Key citation '{cite}' missing from chapter02_related_work.tex"


def test_appendix_d_and_algorithm_1_structure():
    """Verify Appendix D (appendix04.tex) contains tripartite graph definition, bijection, and Algorithm 1."""
    app4_path = MAIN_TEX_DIR / "back" / "appendix04.tex"
    assert app4_path.exists(), "appendix04.tex (Appendix D) does not exist"
    app4_text = app4_path.read_text(encoding="utf-8")

    assert "CSR 雙向局部交集演算法虛擬碼與等價性理論證明" in app4_text
    assert "結構性三部圖拓樸定界與推論通道推導" in app4_text
    assert "演算法虛擬碼" in app4_text
    assert "偽代碼" not in app4_text
    assert "基於 CSR 稀疏矩陣之雙向局部交集路徑萃取演算法" in app4_text
    assert "拓樸雙射與浮點數值等價性定理證明" in app4_text
    assert "alg:csr_bidirectional_search" in app4_text

    main_tex_path = MAIN_TEX_DIR / "main.tex"
    main_tex = main_tex_path.read_text(encoding="utf-8")
    assert "\\input{back/appendix04}" in main_tex, "Appendix D not registered in main.tex"


def test_beamer_and_thesis_baseline_phrasing_consistency():
    """Verify baseline phrasing conforms to ADR-008 (no overclaiming)."""
    defense_tex = (BEAMER_TEX_DIR / "defense.tex").read_text(encoding="utf-8")
    assert "表現最佳之基準模型 LightGCN" in defense_tex
    assert "頂尖基準" not in defense_tex
    assert "頂尖基線" not in defense_tex
    assert "最強基線" not in defense_tex

    abstract_tex = (MAIN_TEX_DIR / "front" / "abstract.tex").read_text(encoding="utf-8")
    assert "表現最佳之基準模型 LightGCN" in abstract_tex
    assert "最強的基準方法" not in abstract_tex
    assert "best-performing baseline LightGCN" in abstract_tex
    assert "strongest baseline" not in abstract_tex

    ch4_tex = (MAIN_TEX_DIR / "contents" / "chapter04_experiments.tex").read_text(encoding="utf-8")
    assert "頂尖基線" not in ch4_tex
    assert "最強基線" not in ch4_tex
    assert "最強 Baseline" not in ch4_tex

    ch5_tex = (MAIN_TEX_DIR / "contents" / "chapter05_conclusion.tex").read_text(encoding="utf-8")
    assert "較表現最佳之基準模型 (LightGCN" in ch5_tex
    assert "最強" not in ch5_tex
