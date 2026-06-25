# ML-модель: предсказываем аномалию урожая по климату.
# Сравниваем OLS / RandomForest / CatBoost.
# Две проверки: обычная случайная CV (как в статьях) и строгая по годам
# (прогноз на годы, которых модель не видела - это честнее, но труднее).
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

import data_io as dio
import viz

PROC = dio.PROC
SEED = 42

# берём только климатические признаки (год и лаг урожая не берём, чтобы мерить
# именно вклад климата). lag1 у SPEI - это прошлый год, не подсматривание.
NUM = ["spei3_gs", "spei6_gs", "spei12_gs", "spi6_gs", "gs_t_z", "gs_p_z",
       "t_max_gs_z", "spei3_min_gs", "spei6_gs_lag1",
       "ann_t", "ann_p", "oni_djf", "oni_mam", "oni_jja"]
CAT = ["iso", "crop"]
LAB = {"spei3_gs": "SPEI-3", "spei6_gs": "SPEI-6", "spei12_gs": "SPEI-12",
       "spi6_gs": "SPI-6", "gs_t_z": "Темп. лета", "gs_p_z": "Осадки лета",
       "t_max_gs_z": "Пик жары", "spei3_min_gs": "Мин. SPEI-3",
       "spei6_gs_lag1": "SPEI-6 (пр. год)", "ann_t": "Темп. год", "ann_p": "Осадки год",
       "oni_djf": "ONI зима", "oni_mam": "ONI весна", "oni_jja": "ONI лето",
       "iso": "Страна", "crop": "Культура"}


def cv_predict(make_model, X, y, splits, cat_idx=None):
    # прогноз на каждом фолде, потом усредняем (для повторной CV)
    preds = np.zeros(len(y))
    counts = np.zeros(len(y))
    for tr, te in splits:
        m = make_model()
        if cat_idx is not None:
            m.fit(X.iloc[tr], y.iloc[tr], cat_features=cat_idx)
        else:
            m.fit(X.iloc[tr], y.iloc[tr])
        preds[te] += m.predict(X.iloc[te])
        counts[te] += 1
    yhat = preds / np.maximum(counts, 1)
    return (r2_score(y, yhat), np.sqrt(mean_squared_error(y, yhat)),
            mean_absolute_error(y, yhat), yhat)


# фабрики моделей (каждый раз новая, чтобы не утекало между фолдами)
def mk_cat():
    return CatBoostRegressor(iterations=500, depth=4, learning_rate=0.04,
                             l2_leaf_reg=6.0, loss_function="RMSE",
                             random_seed=SEED, verbose=False)


def mk_rf():
    return RandomForestRegressor(n_estimators=500, max_depth=7,
                                 min_samples_leaf=5, random_state=SEED, n_jobs=-1)


def mk_ols():
    return LinearRegression()


def main():
    panel = pd.read_csv(os.path.join(PROC, "analysis_panel.csv"))
    df = panel.dropna(subset=NUM + ["yield_anom_pct"]).reset_index(drop=True)
    y = df["yield_anom_pct"]
    print("n=%d наблюдений, признаков: %d числовых + %d категориальных" % (len(df), len(NUM), len(CAT)))

    # два набора разбиений
    yr_splits = list(GroupKFold(n_splits=5).split(df, y, df["year"]))
    rkf_splits = list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(df))

    # для CatBoost категориальные подаём как есть, для RF/OLS делаем one-hot
    Xc = df[NUM + CAT].copy()
    for c in CAT:
        Xc[c] = Xc[c].astype(str)
    cat_idx = [Xc.columns.get_loc(c) for c in CAT]
    Xoh = pd.get_dummies(df[NUM + CAT], columns=CAT)

    res = {"n": int(len(y)), "n_features": len(NUM) + len(CAT), "models": {}}

    # строгая CV (по годам)
    yhat_cat = None
    for name, (mk, X, ci) in {"OLS": (mk_ols, Xoh, None),
                              "RandomForest": (mk_rf, Xoh, None),
                              "CatBoost": (mk_cat, Xc, cat_idx)}.items():
        r2, rmse, mae, yhat = cv_predict(mk, X, y, yr_splits, ci)
        res["models"].setdefault(name, {})["year_blocked"] = {"r2": round(r2, 3), "rmse": round(rmse, 2), "mae": round(mae, 2)}
        if name == "CatBoost":
            yhat_cat = yhat

    # обычная случайная CV
    for name, (mk, X, ci) in {"OLS": (mk_ols, Xoh, None),
                              "RandomForest": (mk_rf, Xoh, None),
                              "CatBoost": (mk_cat, Xc, cat_idx)}.items():
        r2, rmse, mae, _ = cv_predict(mk, X, y, rkf_splits, ci)
        res["models"][name]["random_kfold"] = {"r2": round(r2, 3), "rmse": round(rmse, 2), "mae": round(mae, 2)}

    res["rmse_baseline_zero"] = round(np.sqrt(mean_squared_error(y, np.zeros_like(y))), 2)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    # финальный CatBoost на всех данных - для важности признаков и графиков
    final = mk_cat(); final.fit(Xc, y, cat_features=cat_idx)
    imp = pd.Series(final.get_feature_importance(), index=Xc.columns).sort_values()
    res["feature_importance"] = {k: round(v, 2) for k, v in imp.items()}
    with open(os.path.join(PROC, "ml_results.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # Рис.5 важность признаков
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.barh([LAB.get(k, k) for k in imp.index], imp.values, color=viz.C_ML)
    ax.set_title("Важность признаков (CatBoost)")
    ax.set_xlabel("Вклад в прогноз, %")
    viz.save(fig, "fig05_feature_importance")

    # Рис.6 частные зависимости (как меняется прогноз при изменении одного признака)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, feat, lab in [(axes[0], "spei6_gs", "SPEI-6 (вегетация)"),
                          (axes[1], "t_max_gs_z", "Пик жары лета (z)")]:
        grid = np.linspace(df[feat].quantile(.02), df[feat].quantile(.98), 40)
        preds = []
        for v in grid:
            Xt = Xc.copy(); Xt[feat] = v
            preds.append(final.predict(Xt).mean())
        ax.plot(grid, preds, color=viz.C_DROUGHT, lw=2.2)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(lab); ax.set_ylabel("Прогноз аномалии урожая, %")
        ax.set_title("Частная зависимость: " + lab.split(" (")[0])
    viz.save(fig, "fig06_partial_dependence")

    # Рис.7 прогноз vs факт (вне выборки)
    r2y = res["models"]["CatBoost"]["year_blocked"]["r2"]
    rmsey = res["models"]["CatBoost"]["year_blocked"]["rmse"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y, yhat_cat, s=16, alpha=0.5, color=viz.C_NORMAL)
    lim = [min(y.min(), yhat_cat.min()), max(y.max(), yhat_cat.max())]
    ax.plot(lim, lim, "--", color="grey")
    ax.set_xlabel("Факт. аномалия урожая, %")
    ax.set_ylabel("Прогноз (вне выборки), %")
    ax.set_title("CatBoost, строгая CV по годам\n$R^2$=%.2f, RMSE=%.1f%%" % (r2y, rmsey))
    viz.save(fig, "fig07_pred_vs_actual")
    print("Saved ML figures and ml_results.json")


if __name__ == "__main__":
    main()
