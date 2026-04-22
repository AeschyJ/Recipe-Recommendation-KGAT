"""
generate_figures.py
產生論文所需的兩類圖形：
  1. fig_training_curves.pdf  ── 訓練曲線（HR@20 與 NDCG@20 雙子圖）
  2. fig_ckg_schema.pdf       ── 協同知識圖譜 (CKG) 架構示意圖

執行方式：
  cd Paper/Main/figures
  python generate_figures.py

依賴套件：matplotlib, networkx
  pip install matplotlib networkx
"""

from __future__ import annotations

import pathlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "pdf.fonttype": 42,   # 確保 PDF 中字型可搜尋
    "ps.fonttype": 42,
})

OUT_DIR = pathlib.Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 資料定義（來源：附錄 B 偶數 Epoch 數據）
# ─────────────────────────────────────────────

EPOCHS = [2, 4, 6, 8, 10]

DATA_HR20 = {
    r"KGAT ($L{=}1$)": [0.7449, 0.7690, 0.7801, 0.7862, 0.7910],
    r"KGAT ($L{=}2$)": [0.8369, 0.8564, 0.8666, 0.8725, 0.8766],
    r"KGAT ($L{=}3$)": [0.9120, 0.9241, 0.9324, 0.9348, 0.9368],
}

DATA_NDCG20 = {
    r"KGAT ($L{=}1$)": [0.4033, 0.4232, 0.4345, 0.4400, 0.4440],
    r"KGAT ($L{=}2$)": [0.4830, 0.4993, 0.5087, 0.5132, 0.5165],
    r"KGAT ($L{=}3$)": [0.5608, 0.5764, 0.5848, 0.5871, 0.5891],
}

# Baseline 最終值（用水平虛線標記）
BASELINES = {
    "HR@20":   {"LightGCN": 0.6717, "BPR-MF": 0.5675, "NFM": 0.4476},
    "NDCG@20": {"LightGCN": 0.3858, "BPR-MF": 0.3273, "NFM": 0.2560},
}

# 顏色與線型設定（與論文主色系協調）
LINE_STYLES: dict[str, dict] = {
    r"KGAT ($L{=}1$)": {"color": "#4878CF", "marker": "o", "linestyle": "-"},
    r"KGAT ($L{=}2$)": {"color": "#E07321", "marker": "s", "linestyle": "-"},
    r"KGAT ($L{=}3$)": {"color": "#2CA02C", "marker": "^", "linestyle": "-"},
}

BASELINE_STYLES = {
    "LightGCN": {"color": "#9467BD", "linestyle": "--"},
    "BPR-MF":   {"color": "#8C564B", "linestyle": ":"},
    "NFM":      {"color": "#7F7F7F", "linestyle": "-."},
}


def _plot_metric_axis(
    ax: plt.Axes,
    data: dict[str, list[float]],
    baselines: dict[str, float],
    ylabel: str,
    show_legend: bool = True,
) -> None:
    """在單個 Axes 上繪製一個指標的訓練曲線與 Baseline 基線。"""
    # 主曲線
    for label, values in data.items():
        style = LINE_STYLES[label]
        ax.plot(
            EPOCHS, values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=7,
            label=label,
        )

    # Baseline 水平虛線
    for bl_name, bl_val in baselines.items():
        style = BASELINE_STYLES[bl_name]
        ax.axhline(
            y=bl_val,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.3,
            alpha=0.8,
            label=bl_name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(EPOCHS)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    if show_legend:
        ax.legend(loc="lower right", framealpha=0.85)


def generate_training_curves() -> None:
    """產生 HR@20 與 NDCG@20 的雙子圖訓練曲線。"""
    fig, (ax_hr, ax_ndcg) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

    _plot_metric_axis(ax_hr,   DATA_HR20,   BASELINES["HR@20"],   "HR@20",   show_legend=True)
    _plot_metric_axis(ax_ndcg, DATA_NDCG20, BASELINES["NDCG@20"], "NDCG@20", show_legend=False)

    ax_hr.set_title("(a) HR@20 vs. Epoch")
    ax_ndcg.set_title("(b) NDCG@20 vs. Epoch")

    # 共用圖例放在圖形底部
    handles, labels = ax_hr.get_legend_handles_labels()
    ax_hr.get_legend().remove()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.13),
        framealpha=0.9,
        fontsize=9.5,
    )

    fig.tight_layout()
    out_path = OUT_DIR / "fig_training_curves.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[OK] Training curves saved: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# CKG 架構示意圖
# ─────────────────────────────────────────────

def generate_ckg_schema() -> None:
    """
    繪製協同知識圖譜 (CKG) 架構示意圖。

    節點類型：
      - User（使用者）：深藍圓形
      - Recipe（食譜）：橘色圓形
      - Ingredient（食材實體）：綠色菱形
      - Tag（標籤實體）：紫色菱形

    邊類型：
      - interact / interact⁻¹ ── 使用者-食譜互動
      - has_ingredient / has_ingredient⁻¹ ── 食材關係
      - has_tag / has_tag⁻¹ ── 標籤關係
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # ── 節點位置（手動排版，讓圖形直觀易讀）──
    pos: dict[str, tuple[float, float]] = {
        # 使用者群（左側）
        "u₁": (0.5, 5.0),
        "u₂": (0.5, 3.0),
        # 食譜群（中間）
        "r₁": (3.5, 6.0),
        "r₂": (3.5, 4.0),
        "r₃": (3.5, 2.0),
        # 食材實體（右上）
        "e: salmon":  (6.8, 6.8),
        "e: lemon":   (6.8, 5.5),
        "e: garlic":  (6.8, 4.2),
        # 標籤實體（右下）
        "e: seafood": (6.8, 3.0),
        "e: healthy": (6.8, 1.8),
    }

    # 節點分類
    user_nodes  = ["u₁", "u₂"]
    recipe_nodes = ["r₁", "r₂", "r₃"]
    ingr_nodes  = ["e: salmon", "e: lemon", "e: garlic"]
    tag_nodes   = ["e: seafood", "e: healthy"]

    NODE_COLORS = {
        "user":   "#4878CF",
        "recipe": "#E07321",
        "ingr":   "#2CA02C",
        "tag":    "#9467BD",
    }
    NODE_SIZE = 900

    # ── 繪製邊（先畫邊，後畫節點，避免被遮擋）──
    edges: list[tuple[str, str, str, str]] = [
        # (src, dst, label, anchor_side)
        # interact 邊
        ("u₁", "r₁", "interact", "top"),
        ("u₁", "r₂", "", ""),
        ("u₂", "r₂", "interact", "bottom"),
        ("u₂", "r₃", "", ""),
        # has_ingredient 邊
        ("r₁", "e: salmon", "has_ingredient", "top"),
        ("r₁", "e: lemon",  "", ""),
        ("r₂", "e: lemon",  "", ""),
        ("r₂", "e: garlic", "has_ingredient", "bottom"),
        ("r₃", "e: garlic", "", ""),
        # has_tag 邊
        ("r₁", "e: seafood", "has_tag", "top"),
        ("r₂", "e: healthy", "", ""),
        ("r₃", "e: seafood", "", ""),
        ("r₃", "e: healthy", "has_tag", "bottom"),
    ]

    for src, dst, label, anchor in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        color = "#555555"
        # 邊的顏色依類型
        if "interact" in label or (not label and src in user_nodes):
            color = "#4878CF"
        elif "ingredient" in label or (not label and src in recipe_nodes and dst in ingr_nodes):
            color = "#2CA02C"
        elif "tag" in label or (not label and src in recipe_nodes and dst in tag_nodes):
            color = "#9467BD"

        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.5,
                mutation_scale=14,
                shrinkA=14,
                shrinkB=14,
            ),
        )
        # 反向箭頭（雙向邊）
        ax.annotate(
            "", xy=(x0, y0), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.5,
                mutation_scale=14,
                shrinkA=14,
                shrinkB=14,
            ),
        )
        # 邊標籤（只標第一次出現）
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            offset_y = 0.22 if anchor == "top" else -0.22
            ax.text(
                mx, my + offset_y, label,
                ha="center", va="center",
                fontsize=8.5,
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
            )

    # ── 繪製節點 ──
    def _draw_nodes(names: list[str], color: str, shape: str = "o") -> None:
        xs = [pos[n][0] for n in names]
        ys = [pos[n][1] for n in names]
        ax.scatter(
            xs, ys,
            s=NODE_SIZE,
            c=color,
            marker=shape,
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
        )
        for n in names:
            ax.text(
                pos[n][0], pos[n][1], n,
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white", zorder=6,
            )

    _draw_nodes(user_nodes,  NODE_COLORS["user"],   "o")
    _draw_nodes(recipe_nodes, NODE_COLORS["recipe"], "o")
    _draw_nodes(ingr_nodes,  NODE_COLORS["ingr"],   "D")
    _draw_nodes(tag_nodes,   NODE_COLORS["tag"],    "D")

    # ── 圖例 ──
    legend_elements = [
        mpatches.Patch(color=NODE_COLORS["user"],   label="User node"),
        mpatches.Patch(color=NODE_COLORS["recipe"], label="Recipe node"),
        mpatches.Patch(color=NODE_COLORS["ingr"],   label="Ingredient entity"),
        mpatches.Patch(color=NODE_COLORS["tag"],    label="Tag entity"),
        mlines.Line2D([], [], color=NODE_COLORS["user"],   lw=1.5,
                      label="interact / interact⁻¹"),
        mlines.Line2D([], [], color=NODE_COLORS["ingr"],   lw=1.5,
                      label="has_ingredient / has_ingredient⁻¹"),
        mlines.Line2D([], [], color=NODE_COLORS["tag"],    lw=1.5,
                      label="has_tag / has_tag⁻¹"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        fontsize=9,
        framealpha=0.9,
        ncol=2,
    )

    # ── 區域標籤 ──
    for label, x, y in [
        ("User\nGraph", 0.5, 6.3),
        ("Knowledge\nGraph", 6.8, 7.5),
        ("Collaborative\nKnowledge Graph (CKG)", 3.5, 7.5),
    ]:
        ax.text(x, y, label, ha="center", va="bottom",
                fontsize=10, fontstyle="italic", color="#444444")

    # 用大框標示 CKG 涵蓋範圍
    rect = mpatches.FancyBboxPatch(
        (0.0, 1.2), 7.8, 6.6,
        boxstyle="round,pad=0.1",
        linewidth=1.5, edgecolor="#BBBBBB",
        facecolor="none", linestyle="--", zorder=0,
    )
    ax.add_patch(rect)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(0.8, 8.2)

    out_path = OUT_DIR / "fig_ckg_schema.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[OK] CKG schema saved: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main() -> None:
    print("=== Generating figures ===")
    generate_training_curves()
    generate_ckg_schema()
    print("=== Done ===")
    print()
    print("LaTeX usage:")
    print(r"  \includegraphics[width=0.9\linewidth]{figures/fig_training_curves.pdf}")
    print(r"  \includegraphics[width=0.75\linewidth]{figures/fig_ckg_schema.pdf}")


if __name__ == "__main__":
    main()
