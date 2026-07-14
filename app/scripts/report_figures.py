"""
report_figures.py
Unified tumour-analysis figure for the app and notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LABEL_NCR, LABEL_ED, LABEL_ET = 1, 2, 4

# Disjoint colour scheme
COL_ET = (1.00, 0.00, 0.00)   # red  
COL_NCR = (0.00, 0.40, 1.00)   # blue  
COL_ED = (1.00, 0.90, 0.00)   # yellow 


def _norm01(v):
    v = v.astype(np.float32)
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    return np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(v)


def _key_slice(seg):
    tps = (seg > 0).sum(axis=(0, 1))
    return int(tps.argmax()) if tps.sum() > 0 else seg.shape[2] // 2


# overlay disjoint
def draw_overlay_disjoint(ax, base_vol, seg, z=None, alpha=0.6):
    """Segmentation overlay."""
    z = _key_slice(seg) if z is None else z
    b2d = np.rot90(_norm01(base_vol[:, :, z]))
    s2d = np.rot90(seg[:, :, z])
    rgba = np.zeros((*s2d.shape, 4))
    for lab, col in [(LABEL_ET, COL_ET), (LABEL_NCR, COL_NCR), (LABEL_ED, COL_ED)]:
        rgba[s2d == lab] = [*col, alpha]
    ax.imshow(b2d, cmap="gray")
    ax.imshow(rgba, interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"Segmentation Overlay\n(Slice {z})", fontsize=14, fontweight="bold")
    patches = [mpatches.Patch(color=COL_ET,  label="Enhancing (ET)"),
               mpatches.Patch(color=COL_NCR, label="Necrotic (NCR)"),
               mpatches.Patch(color=COL_ED,  label="Edema (ED)")]
    ax.legend(handles=patches, loc="lower left", fontsize=11, framealpha=0.8)


# donut disjoint
def draw_donut_disjoint(ax, et, ncr, ed):
    """Disjoint composition donut (sums to 100%)."""
    vals = [v if (v and not np.isnan(v)) else 0.0 for v in (et, ncr, ed)]
    colors = [COL_ET, COL_NCR, COL_ED]
    labels = ["Enhancing (ET)", "Necrotic (NCR)", "Edema (ED)"]
    if sum(vals) > 0:
        wedges, _, autot = ax.pie(vals, colors=colors, autopct="%1.1f%%", startangle=90,
                                  pctdistance=0.78, wedgeprops=dict(width=0.45, edgecolor="white"))
        plt.setp(autot, size=12, weight="bold")
        ax.legend(wedges, [f"{l}: {v:.1f} cm³" for l, v in zip(labels, vals)],
                  loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=12)
    ax.set_title("Tissue Composition (Disjoint)", fontsize=14, fontweight="bold")


# nested bars
def draw_nested_bars(ax, et_v, tc_v, wt_v):
    vals = [wt_v, tc_v, et_v]
    labels = ["WT (Whole)", "TC (Core)", "ET (Enhancing)"]
    
    colors = [(0.4, 0.45, 0.5), (0.45, 0.3, 0.45), (1.0, 0.0, 0.0)]
    
    ypos = np.arange(3)[::-1]
    ax.barh(ypos, vals, color=colors, edgecolor='black', alpha=0.85)
    
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontweight='bold')
    ax.set_title("Clinical Hierarchy\n(ET ⊆ TC ⊆ WT)", fontsize=12, fontweight='bold')
    
    # Add volume labels at the end of bars
    span = max(vals) if max(vals) > 0 else 1.0
    for i, v in enumerate(vals):
        ax.text(v + span * 0.02, ypos[i], f'{v:.1f} cm³', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, span * 1.2)
    ax.set_xlabel("Volume (cm³)")


# unified figure
def figure_unified_summary(base_vol, seg, et_cm3, ncr_cm3, ed_cm3, pid=""):
    """One row: segmentation overlay | composition donut | nested-region bars.
    All three use the same disjoint colour language (red/blue/yellow)."""
    tc_cm3 = et_cm3 + ncr_cm3
    wt_cm3 = tc_cm3 + ed_cm3
    fig, (ax_img, ax_pie, ax_bar) = plt.subplots(1, 3, figsize=(22, 8))
    draw_overlay_disjoint(ax_img, base_vol, seg)
    draw_donut_disjoint(ax_pie, et_cm3, ncr_cm3, ed_cm3)
    draw_nested_bars(ax_bar, et_cm3, tc_cm3, wt_cm3)
    fig.suptitle(f"Tumour Analysis - Patient ID: {pid}", fontsize=17, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

