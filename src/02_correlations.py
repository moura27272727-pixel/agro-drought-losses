# Корреляции (базовый уровень) + разведочные графики.
# Считаем связь аномалии урожая с засушливостью: по России и по панели (с FE).
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import data_io as dio
import viz

PROC = dio.PROC
DROUGHT_YEARS_RU = [1998, 2010, 2012, 2021]   # засушливые годы в РФ (подсветим на графике)


def corr(a, b):
    # корреляции Пирсона и Спирмена с выкидыванием NaN
    m = ~(np.isnan(a) | np.isnan(b))
    r, p = stats.pearsonr(a[m], b[m])
    rs, ps = stats.spearmanr(a[m], b[m])
    return r, p, rs, ps, int(m.sum())


def main():
    panel = pd.read_csv(os.path.join(PROC, "analysis_panel.csv"))
    ru = pd.read_csv(os.path.join(PROC, "russia_annual.csv")).sort_values("year")

    metrics = {
        "spei6_gs": "SPEI-6 (вег.)", "spei12_gs": "SPEI-12 (вег.)",
        "spi6_gs": "SPI-6 (вег.)", "gs_t_z": "Темп. лета (z)",
        "gs_p_z": "Осадки лета (z)", "oni_jja": "ONI (лето)",
    }

    rows = []
    # --- Россия, пшеница ---
    y = ru["yield_anom_pct"].values
    for col, name in metrics.items():
        x = ru[col].values.astype(float)
        sign = -1 if col in ("gs_t_z",) else 1   # жара со знаком минус, ждём + корреляцию
        r, p, rs, ps, n = corr(y, sign * x)
        rows.append({"scope": "RUS_wheat", "metric": name, "n": n,
                     "pearson": round(r, 2), "p": round(p, 3), "spearman": round(rs, 2)})

    # --- панель по странам (пшеница), внутри стран (вычли страновые средние = FE) ---
    w = panel[panel.crop == "Wheat"].copy()
    for col, name in metrics.items():
        ww = w.dropna(subset=["yield_anom_pct", col]).copy()
        sign = -1 if col in ("gs_t_z",) else 1
        ww["yw"] = ww["yield_anom_pct"] - ww.groupby("iso")["yield_anom_pct"].transform("mean")
        ww["xw"] = sign * (ww[col] - ww.groupby("iso")[col].transform("mean"))
        r, p, rs, ps, n = corr(ww["yw"].values, ww["xw"].values)
        rows.append({"scope": "Panel_wheat_FE", "metric": name, "n": n,
                     "pearson": round(r, 2), "p": round(p, 4), "spearman": round(rs, 2)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "results_correlations.csv"), index=False)
    print(res.to_string(index=False))

    # ====== графики ======
    # Рис.1 урожай пшеницы РФ + тренд
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(ru.year, ru["yield"] / 1000, "o-", color=viz.C_NORMAL, lw=1.6, ms=4, label="Урожайность")
    ax.plot(ru.year, ru["yield_trend"] / 1000, "--", color=viz.C_TREND, lw=2,
            label="Технологический тренд (LOESS)")
    for yr in DROUGHT_YEARS_RU:
        if yr in ru.year.values:
            ax.axvspan(yr - 0.4, yr + 0.4, color=viz.C_DROUGHT, alpha=0.12)
    ax.set_title("Урожайность пшеницы в России и засушливые годы")
    ax.set_xlabel("Год"); ax.set_ylabel("Урожайность, т/га")
    ax.legend(frameon=False, fontsize=9)
    viz.save(fig, "fig01_russia_yield_trend")

    # Рис.2 SPEI vs аномалия урожая (РФ)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    x = ru["spei6_gs"].values; yv = ru["yield_anom_pct"].values
    m = ~(np.isnan(x) | np.isnan(yv))
    ax.scatter(x[m], yv[m], color=viz.C_NORMAL, s=28, zorder=3)
    b = np.polyfit(x[m], yv[m], 1)   # линия тренда
    xs = np.linspace(x[m].min(), x[m].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=viz.C_DROUGHT, lw=2)
    for _, r in ru.iterrows():
        if int(r.year) in (2010, 2012, 2021):
            ax.annotate(int(r.year), (r.spei6_gs, r.yield_anom_pct),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
    rr, pp, _, _, _ = corr(ru["spei6_gs"].values, ru["yield_anom_pct"].values)
    ax.set_title("Россия: SPEI-6 и аномалия урожая\nr=%.2f, p=%.3f" % (rr, pp))
    ax.set_xlabel("SPEI-6 (конец вегетации)"); ax.set_ylabel("Аномалия урожая пшеницы, %")
    ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
    viz.save(fig, "fig02_russia_spei_yield_scatter")

    # Рис.3 корреляции по странам
    countries = list(dio.COUNTRIES)
    rs_spei, rs_temp = [], []
    for iso in countries:
        g = panel[(panel.iso == iso) & (panel.crop == "Wheat")]
        r1, _, _, _, _ = corr(g["yield_anom_pct"].values, g["spei6_gs"].values)
        r2, _, _, _, _ = corr(g["yield_anom_pct"].values, -g["gs_t_z"].values)
        rs_spei.append(r1); rs_temp.append(r2)
    xpos = np.arange(len(countries)); wbar = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(xpos - wbar / 2, rs_spei, wbar, color=viz.C_NORMAL, label="SPEI-6")
    ax.bar(xpos + wbar / 2, rs_temp, wbar, color=viz.C_DROUGHT, label="−Темп. лета")
    ax.set_xticks(xpos)
    ax.set_xticklabels([dio.COUNTRIES[c]["ru"] for c in countries])
    ax.set_ylabel("Коэф. корреляции (Пирсон)")
    ax.set_title("Связь засушливости и аномалии урожая пшеницы по странам")
    ax.legend(frameon=False, fontsize=9); ax.axhline(0, color="grey", lw=0.8)
    viz.save(fig, "fig03_country_corr_bars")

    # Рис.4 временной ряд SPEI-6 (РФ)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    s = ru["spei6_gs"].values
    ax.bar(ru.year, s, color=[viz.C_DROUGHT if v < 0 else viz.C_NORMAL for v in s])
    ax.axhline(-1, color="black", lw=0.8, ls=":")
    ax.text(ru.year.min(), -1.05, "порог засухи SPEI=−1", fontsize=8, va="top")
    ax.set_title("Индекс засухи SPEI-6 (конец вегетации), Россия")
    ax.set_xlabel("Год"); ax.set_ylabel("SPEI-6")
    viz.save(fig, "fig04_russia_spei_timeseries")


if __name__ == "__main__":
    main()
