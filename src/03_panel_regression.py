# Панельные регрессии с фиксированными эффектами (это уже сложнее корреляций).
# Три модели: A) пшеница, FE стран; B) все культуры, FE страна x культура;
# C) пшеница, FE стран + годов. SE считаем по Дрисколлу-Краая.
import json
import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

import data_io as dio

PROC = dio.PROC
TABLES = os.path.join(dio.ROOT, "report", "tables")
os.makedirs(TABLES, exist_ok=True)
REGRS = ["spei6_gs", "gs_t_z", "oni_jja"]
LABELS = {"spei6_gs": "SPEI-6 (вегетация)", "gs_t_z": "Темп. лета (z)", "oni_jja": "ONI (лето)"}


def fit(data, entity_col, regressors, time_effects=False):
    d = data.dropna(subset=regressors + ["yield_anom_pct"]).copy()
    d = d.set_index([entity_col, "year"])
    formula = "yield_anom_pct ~ " + " + ".join(regressors) + " + EntityEffects"
    if time_effects:
        formula += " + TimeEffects"
    mod = PanelOLS.from_formula(formula, data=d, drop_absorbed=True)
    # SE Дрисколла-Краая: устойчивы к зависимости между странами (общий ENSO)
    # и к маленькому числу панельных единиц - честнее, чем кластеры по 6 странам
    return mod.fit(cov_type="kernel", kernel="bartlett")


def summarize(res, name):
    out = {"model": name, "n": int(res.nobs), "r2_within": float(res.rsquared_within), "coef": {}}
    for k in res.params.index:
        out["coef"][k] = {"b": float(res.params[k]), "se": float(res.std_errors[k]),
                          "p": float(res.pvalues[k])}
    return out


def main():
    panel = pd.read_csv(os.path.join(PROC, "analysis_panel.csv"))
    wheat = panel[panel.crop == "Wheat"].copy()
    allc = panel.copy()
    allc["unit"] = allc["iso"] + "_" + allc["crop"]   # единица панели для модели B

    results = []
    print("=== A. Пшеница, FE стран ===")
    rA = fit(wheat, "iso", REGRS)
    print(rA.params.round(3).to_string()); print("within R2:", round(rA.rsquared_within, 3))
    results.append(summarize(rA, "A_wheat_countryFE"))

    print("\n=== B. Все культуры, FE страна×культура ===")
    rB = fit(allc, "unit", REGRS)
    print(rB.params.round(3).to_string()); print("within R2:", round(rB.rsquared_within, 3))
    results.append(summarize(rB, "B_allcrops_unitFE"))

    print("\n=== C. Пшеница, FE стран + годов ===")
    rC = fit(wheat, "iso", REGRS, time_effects=True)
    print(rC.params.round(3).to_string()); print("within R2:", round(rC.rsquared_within, 3))
    results.append(summarize(rC, "C_wheat_country+yearFE"))

    with open(os.path.join(PROC, "panel_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # собираем таблицу для отчёта (LaTeX). Звёздочки - уровни значимости.
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r" & (A) Пшеница & (B) Все культуры & (C) Пшеница \\",
             r" & FE стран & FE страна$\times$культура & FE стран+годов \\",
             r"\midrule"]
    for k in REGRS:
        row = LABELS[k]
        for res in (rA, rB, rC):
            if k in res.params.index:
                b, se, p = res.params[k], res.std_errors[k], res.pvalues[k]
                st = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
                star = f"$^{{{st}}}$" if st else ""
                row += f" & {b:.2f}{star} ({se:.2f})"
            else:
                row += " & --"
        lines.append(row + r" \\")
    lines += [r"\midrule",
              f"Набл. & {int(rA.nobs)} & {int(rB.nobs)} & {int(rC.nobs)} \\\\",
              f"$R^2$ (внутр.) & {rA.rsquared_within:.2f} & {rB.rsquared_within:.2f} & {rC.rsquared_within:.2f} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLES, "tab_panel.tex"), "w") as f:
        f.write("\n".join(lines))
    print("\nSaved: report/tables/tab_panel.tex; data/processed/panel_results.json")


if __name__ == "__main__":
    main()
