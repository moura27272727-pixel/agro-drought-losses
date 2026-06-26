# единый стиль графиков; сохранение в pdf и png
import os
import matplotlib
matplotlib.use("Agg")          # backend без дисплея
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # шрифт с кириллицей
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

# палитра
C_DROUGHT = "#b5341f"
C_NORMAL = "#2b6a8f"
C_TREND = "#d68a1e"
C_ML = "#3a7d44"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  saved:", name)
