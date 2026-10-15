"""Cartoon figures for the Exam 2 practice explanations.

Regenerate: python -m pchemapps.exam2_figs
"""
from pathlib import Path

import numpy as np

FIG_DIR = Path(__file__).parent / "assets" / "exam2_figs"


def path(name: str) -> Path:
    return FIG_DIR / name


def rev_vs_irrev_pv():
    """Isothermal expansion: area under the isotherm vs. the P_ext rectangle (not to scale)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vi, vf, pf = 1.0, 6.0, 1.0
    v = np.linspace(vi, vf, 200)
    p = pf * vf / v

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.fill_between(v, p, color="tab:blue", alpha=0.25, lw=0)
    ax.fill_between([vi, vf], pf, color="tab:orange", alpha=0.6, lw=0)
    ax.plot(v, p, color="tab:blue", lw=2)
    ax.plot([vi, vi, vf], [pf * vf / vi, pf, pf], color="tab:orange", lw=2, ls="--")
    ax.plot([vi, vf], [pf * vf / vi, pf], "ko", ms=5)
    ax.annotate("start", (vi, pf * vf / vi), xytext=(8, 0), textcoords="offset points", va="center")
    ax.annotate("end", (vf, pf), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.text(2.0, 3.6, "reversible:\n$P_\\mathrm{ext}$ = $P_\\mathrm{gas}$ the whole way\n"
            "$-w$ = whole area under the curve\n(blue + orange)", color="tab:blue")
    ax.text(3.5, 0.45, "irreversible: $-w = P_\\mathrm{ext}\\,\\Delta V$", color="black", ha="center")
    ax.set_xlabel("$V$")
    ax.set_ylabel("$P$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, vf + 0.5)
    ax.set_ylim(0, pf * vf / vi + 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Same start and end, constant $T$ (not to scale)", fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path("rev_vs_irrev_pv.png"))
    plt.close(fig)


if __name__ == "__main__":
    rev_vs_irrev_pv()
