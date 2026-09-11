"""
generate_barchart.py
繪製 Food.com 食譜資料集上各模型之推薦效能對比長條圖 (HR@20 與 NDCG@20)
產出：fig_performance_barchart.pdf
"""

import pathlib
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIR = pathlib.Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

models = ["BPR-MF", "NFM", "LightGCN", "KGAT-1L", "KGAT-2L", "KGAT-3L\n(Ours)"]
hr20_scores = [0.5830, 0.6313, 0.6668, 0.6761, 0.7711, 0.8775]
ndcg20_scores = [0.3355, 0.3712, 0.3836, 0.3749, 0.4383, 0.5309]

colors = [
    "#90A4AE",  # BPR-MF
    "#78909C",  # NFM
    "#546E7A",  # LightGCN (Best Baseline)
    "#42A5F5",  # KGAT-1L
    "#1E88E5",  # KGAT-2L
    "#C00000",  # KGAT-3L (Highlight Red)
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- HR@20 ---
bars1 = ax1.bar(models, hr20_scores, color=colors, width=0.6, edgecolor="black", linewidth=0.6)
ax1.set_title("(a) HR@20 Comparison (Higher is Better)")
ax1.set_ylim(0.4, 0.98)
ax1.set_ylabel("HR@20")
ax1.grid(axis="y", linestyle=":", alpha=0.6)

for bar, score in zip(bars1, hr20_scores):
    height = bar.get_height()
    fontweight = "bold" if score == max(hr20_scores) else "normal"
    color = "#C00000" if score == max(hr20_scores) else "black"
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.012,
        f"{score:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight=fontweight,
        color=color,
    )

# +31.6% 標註 (KGAT-3L vs LightGCN)
ax1.annotate(
    "+31.6%",
    xy=(5, 0.8775),
    xytext=(3.4, 0.915),
    arrowprops=dict(arrowstyle="->", color="#C00000", lw=1.5),
    fontsize=10.5,
    fontweight="bold",
    color="#C00000",
    bbox=dict(boxstyle="round,pad=0.2", fc="#FFEBEE", ec="#C00000", lw=1),
)

# --- NDCG@20 ---
bars2 = ax2.bar(models, ndcg20_scores, color=colors, width=0.6, edgecolor="black", linewidth=0.6)
ax2.set_title("(b) NDCG@20 Comparison (Higher is Better)")
ax2.set_ylim(0.25, 0.60)
ax2.set_ylabel("NDCG@20")
ax2.grid(axis="y", linestyle=":", alpha=0.6)

for bar, score in zip(bars2, ndcg20_scores):
    height = bar.get_height()
    fontweight = "bold" if score == max(ndcg20_scores) else "normal"
    color = "#C00000" if score == max(ndcg20_scores) else "black"
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.008,
        f"{score:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight=fontweight,
        color=color,
    )

# +38.4% 標註 (KGAT-3L vs LightGCN)
ax2.annotate(
    "+38.4%",
    xy=(5, 0.5309),
    xytext=(3.4, 0.555),
    arrowprops=dict(arrowstyle="->", color="#C00000", lw=1.5),
    fontsize=10.5,
    fontweight="bold",
    color="#C00000",
    bbox=dict(boxstyle="round,pad=0.2", fc="#FFEBEE", ec="#C00000", lw=1),
)

fig.tight_layout()
out_path = OUT_DIR / "fig_performance_barchart.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"[OK] Bar chart saved to: {out_path}")
plt.close(fig)
