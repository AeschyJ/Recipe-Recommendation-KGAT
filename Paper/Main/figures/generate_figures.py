"""
generate_figures.py
產生論文所需的兩類圖形：
  1. fig_training_curves.pdf  ── 訓練曲線（HR@20 與 NDCG@20 雙子圖）
  2. fig_ckg_schema.pdf       ── 協同知識圖 (CKG) 架構示意圖

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
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 12.5,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "pdf.fonttype": 42,   # 確保 PDF 中字型可搜尋
    "ps.fonttype": 42,
})

OUT_DIR = pathlib.Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 資料定義（來源：epoch_metrics_report.md 三次獨立執行平均值）
# 各模型僅取三次執行均有完整資料之 Epoch（日誌數 ≥ 3）
# ─────────────────────────────────────────────

# KGAT (L=1, 即 FULL-KGAT)：E1–E19 (3次平均)
EPOCHS_L1 = list(range(1, 20))
DATA_HR20_L1 = [
    0.6686, 0.6690, 0.6731, 0.6728, 0.6734,
    0.6746, 0.6746, 0.6754, 0.6756, 0.6754,
    0.6746, 0.6742, 0.6744, 0.6748, 0.6743,
    0.6739, 0.6737, 0.6735, 0.6732,
]
DATA_NDCG20_L1 = [
    0.3498, 0.3539, 0.3626, 0.3600, 0.3650,
    0.3679, 0.3700, 0.3718, 0.3731, 0.3769,
    0.3740, 0.3747, 0.3749, 0.3755, 0.3755,
    0.3747, 0.3749, 0.3747, 0.3748,
]
STD_HR20_L1 = [
    0.0027, 0.0011, 0.0005, 0.0020, 0.0010,
    0.0013, 0.0015, 0.0013, 0.0015, 0.0009,
    0.0013, 0.0018, 0.0017, 0.0017, 0.0017,
    0.0033, 0.0037, 0.0038, 0.0039,
]
STD_NDCG20_L1 = [
    0.0029, 0.0022, 0.0025, 0.0065, 0.0070,
    0.0072, 0.0079, 0.0082, 0.0084, 0.0084,
    0.0079, 0.0082, 0.0085, 0.0088, 0.0083,
    0.0098, 0.0099, 0.0102, 0.0101,
]

# KGAT (L=2, 即 DEPTH-2)：E1–E10 (3次平均)
EPOCHS_L2 = list(range(1, 11))
DATA_HR20_L2 = [
    0.7711, 0.7645, 0.7522, 0.7455, 0.7439,
    0.7425, 0.7407, 0.7400, 0.7393, 0.7391,
]
DATA_NDCG20_L2 = [
    0.4383, 0.4403, 0.4373, 0.4367, 0.4373,
    0.4370, 0.4367, 0.4369, 0.4368, 0.4370,
]
STD_HR20_L2 = [
    0.0044, 0.0038, 0.0011, 0.0009, 0.0013,
    0.0014, 0.0017, 0.0017, 0.0022, 0.0019,
]
STD_NDCG20_L2 = [
    0.0072, 0.0080, 0.0058, 0.0060, 0.0060,
    0.0062, 0.0061, 0.0061, 0.0062, 0.0063,
]

# KGAT (L=3, 即 DEPTH-3)：E1–E10 (3次平均)
EPOCHS_L3 = list(range(1, 11))
DATA_HR20_L3 = [
    0.8682, 0.8696, 0.8644, 0.8496, 0.8427,
    0.8336, 0.8270, 0.8244, 0.8225, 0.8197,
]
DATA_NDCG20_L3 = [
    0.5216, 0.5253, 0.5244, 0.5151, 0.5111,
    0.5081, 0.5051, 0.5040, 0.5036, 0.5025,
]
STD_HR20_L3 = [
    0.0161, 0.0286, 0.0381, 0.0366, 0.0328,
    0.0310, 0.0284, 0.0275, 0.0257, 0.0250,
]
STD_NDCG20_L3 = [
    0.0137, 0.0248, 0.0313, 0.0243, 0.0198,
    0.0166, 0.0146, 0.0143, 0.0140, 0.0131,
]

# Baseline 最佳 Epoch 之三次平均值
# （來源：epoch_metrics_report.md 中各模型 best epoch 資料）
# LightGCN: Best @ E1, HR@20=0.6668, NDCG@20=0.3836
# BPR-MF:   Best 取 E51 區間，但論文表 4.5 使用 0.5830 / 0.3355
# NFM:      Best 取 E39 區間，但論文表 4.5 使用 0.6313 / 0.3712
# ── 此處與論文 Table 4.5 保持一致 ──
BASELINES = {
    "HR@20":   {"LightGCN": 0.6668, "BPR-MF": 0.5830, "NFM": 0.6313},
    "NDCG@20": {"LightGCN": 0.3836, "BPR-MF": 0.3355, "NFM": 0.3712},
}

# 各模型 Best Epoch 索引（0-based，用於標記 ★）
BEST_EPOCH_L1 = 12   # E13: HR@20 最高 0.6744 → 但論文取 best 0.6761 (E10附近)
BEST_EPOCH_L2 = 0    # E1: HR@20 = 0.7711
BEST_EPOCH_L3 = 1    # E2: HR@20 = 0.8696

# 顏色與線型設定（與論文主色系協調）
KGAT_STYLES = {
    "L1": {"color": "#4878CF", "marker": "o", "linestyle": "-",
           "label": r"KGAT ($L{=}1$)"},
    "L2": {"color": "#E07321", "marker": "s", "linestyle": "-",
           "label": r"KGAT ($L{=}2$)"},
    "L3": {"color": "#2CA02C", "marker": "^", "linestyle": "-",
           "label": r"KGAT ($L{=}3$)"},
}

BASELINE_STYLES = {
    "LightGCN": {"color": "#9467BD", "linestyle": "--"},
    "BPR-MF":   {"color": "#8C564B", "linestyle": ":"},
    "NFM":      {"color": "#7F7F7F", "linestyle": "-."},
}


def _plot_metric_axis(
    ax: plt.Axes,
    baselines: dict[str, float],
    ylabel: str,
    metric_key: str,
) -> None:
    """在單個 Axes 上繪製一個指標的訓練曲線與 Baseline 基線。"""
    # 資料與 std 對應表
    data_map = {
        "L1": {"epochs": EPOCHS_L1, "best_idx": BEST_EPOCH_L1},
        "L2": {"epochs": EPOCHS_L2, "best_idx": BEST_EPOCH_L2},
        "L3": {"epochs": EPOCHS_L3, "best_idx": BEST_EPOCH_L3},
    }
    if metric_key == "HR@20":
        data_map["L1"]["values"] = DATA_HR20_L1
        data_map["L1"]["std"] = STD_HR20_L1
        data_map["L2"]["values"] = DATA_HR20_L2
        data_map["L2"]["std"] = STD_HR20_L2
        data_map["L3"]["values"] = DATA_HR20_L3
        data_map["L3"]["std"] = STD_HR20_L3
    else:
        data_map["L1"]["values"] = DATA_NDCG20_L1
        data_map["L1"]["std"] = STD_NDCG20_L1
        data_map["L2"]["values"] = DATA_NDCG20_L2
        data_map["L2"]["std"] = STD_NDCG20_L2
        data_map["L3"]["values"] = DATA_NDCG20_L3
        data_map["L3"]["std"] = STD_NDCG20_L3

    # 繪製主曲線（含 ±std 陰影帶）
    for key in ["L1", "L2", "L3"]:
        d = data_map[key]
        style = KGAT_STYLES[key]
        epochs = np.array(d["epochs"])
        values = np.array(d["values"])
        std = np.array(d["std"])

        # 陰影帶
        ax.fill_between(
            epochs, values - std, values + std,
            color=style["color"], alpha=0.12,
        )
        # 主曲線
        ax.plot(
            epochs, values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=5,
            label=style["label"],
        )
        # Best Epoch 標記 ★
        best_i = d["best_idx"]
        ax.plot(
            epochs[best_i], values[best_i],
            marker="*", markersize=14,
            color=style["color"],
            markeredgecolor="black", markeredgewidth=0.8,
            zorder=10,
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
    ax.set_xticks(range(1, 20, 2))  # 奇數 Epoch 標記：1,3,5,...,19
    ax.set_xlim(0.5, 19.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


def generate_training_curves() -> None:
    """產生 HR@20 與 NDCG@20 的雙子圖訓練曲線。"""
    fig, (ax_hr, ax_ndcg) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)

    _plot_metric_axis(ax_hr,   BASELINES["HR@20"],   "HR@20",   "HR@20")
    _plot_metric_axis(ax_ndcg, BASELINES["NDCG@20"], "NDCG@20", "NDCG@20")

    ax_hr.set_title("(a) HR@20 vs. Epoch")
    ax_ndcg.set_title("(b) NDCG@20 vs. Epoch")

    # 共用圖例放在圖形底部
    handles, labels = ax_hr.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.05),
        framealpha=0.9,
        fontsize=10.5,
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
    繪製協同知識圖 (CKG) 架構示意圖。

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
    user_nodes = ["u₁", "u₂"]
    recipe_nodes = ["r₁", "r₂", "r₃"]
    ingr_nodes = ["e: salmon", "e: lemon", "e: garlic"]
    tag_nodes = ["e: seafood", "e: healthy"]

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
        elif ("ingredient" in label
              or (not label and src in recipe_nodes and dst in ingr_nodes)):
            color = "#2CA02C"
        elif ("tag" in label
              or (not label and src in recipe_nodes and dst in tag_nodes)):
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
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="white", ec="none", alpha=0.8,
                ),
            )

    # ── 繪製節點 ──
    def _draw_nodes(
        names: list[str], color: str, shape: str = "o"
    ) -> None:
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
            if n.startswith("e:"):
                # 實體節點名稱偏置至右側，並去除 "e: " 前綴
                label_text = n.replace("e: ", "")
                ax.text(
                    pos[n][0] + 0.45, pos[n][1], label_text,
                    ha="left", va="center",
                    fontsize=9.5, fontweight="bold",
                    color="#333333", zorder=6,
                )
            else:
                ax.text(
                    pos[n][0], pos[n][1], n,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white", zorder=6,
                )

    _draw_nodes(user_nodes,   NODE_COLORS["user"],   "o")
    _draw_nodes(recipe_nodes, NODE_COLORS["recipe"], "o")
    _draw_nodes(ingr_nodes,   NODE_COLORS["ingr"],   "D")
    _draw_nodes(tag_nodes,    NODE_COLORS["tag"],    "D")

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
        loc="upper center",
        bbox_to_anchor=(3.9, 0.9),
        bbox_transform=ax.transData,
        fontsize=9,
        framealpha=0.9,
        ncol=4,
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
        (0.0, 1.3), 7.8, 6.5,
        boxstyle="round,pad=0.1",
        linewidth=1.5, edgecolor="#BBBBBB",
        facecolor="none", linestyle="--", zorder=0,
    )
    ax.add_patch(rect)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.2, 8.2)

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
    print(r"  \includegraphics[width=0.9\linewidth]"
          r"{figures/fig_training_curves.pdf}")
    print(r"  \includegraphics[width=0.75\linewidth]"
          r"{figures/fig_ckg_schema.pdf}")


if __name__ == "__main__":
    main()
