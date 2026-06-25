# Собираем главную таблицу для анализа: страна x год x культура.
# Climate-признаки (SPEI/SPI, аномалии температуры и осадков, ENSO) + аномалии урожая.
# Запуск: python src/01_build_panel.py
import os
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

import data_io as dio
from spei import thornthwaite_pet, spi, spei

PROC = dio.PROC
BASE = (1991, 2020)   # период, относительно которого считаем аномалии


def climate_features_for(iso):
    # считаем климатические признаки по одной стране
    d = dio.load_climate(iso)
    lat = dio.COUNTRIES[iso]["lat"]
    months = d["month"].values
    d["pet"] = thornthwaite_pet(d["tas"].values, months, lat)
    d["D"] = d["pr"] - d["pet"]   # водный баланс
    # SPEI разных масштабов + SPI (калибруем по всему ряду 1901-2022)
    for sc in (3, 6, 12):
        d["spei%d" % sc] = spei(d["D"].values, months, sc)
    d["spi6"] = spi(d["pr"].values, months, 6)
    d["spi12"] = spi(d["pr"].values, months, 12)

    # берём значения на август (конец вегетации) и на декабрь (за весь год)
    aug = d[d.month == 8].set_index("year")
    dec = d[d.month == 12].set_index("year")
    feats = pd.DataFrame(index=sorted(d.year.unique()))
    feats["spei3_gs"] = aug["spei3"]
    feats["spei6_gs"] = aug["spei6"]
    feats["spei12_gs"] = aug["spei12"]
    feats["spi6_gs"] = aug["spi6"]
    feats["spei12_ann"] = dec["spei12"]

    # лето (май-август) и год: температура и осадки, плюс их аномалии (z-оценки)
    gs = d[d.month.between(5, 8)].groupby("year").agg(gs_t=("tas", "mean"), gs_p=("pr", "sum"))
    ann = d.groupby("year").agg(ann_t=("tas", "mean"), ann_p=("pr", "sum"))
    base = (d.year >= BASE[0]) & (d.year <= BASE[1])
    gs_b = d[base & d.month.between(5, 8)].groupby("year").agg(t=("tas", "mean"), p=("pr", "sum"))
    feats["gs_t"] = gs["gs_t"]
    feats["gs_p"] = gs["gs_p"]
    feats["gs_t_z"] = (gs["gs_t"] - gs_b["t"].mean()) / gs_b["t"].std()
    feats["gs_p_z"] = (gs["gs_p"] - gs_b["p"].mean()) / gs_b["p"].std()
    feats["ann_t"] = ann["ann_t"]
    feats["ann_p"] = ann["ann_p"]
    # ещё два признака: пиковая жара лета и самый сухой месяц сезона
    tmax = d[d.month.between(5, 8)].groupby("year")["tas"].max()
    tmax_b = d[base & d.month.between(5, 8)].groupby("year")["tas"].max()
    feats["t_max_gs_z"] = (tmax - tmax_b.mean()) / tmax_b.std()
    feats["spei3_min_gs"] = d[d.month.between(4, 8)].groupby("year")["spei3"].min()

    feats = feats.reset_index().rename(columns={"index": "year"})
    feats.insert(0, "iso", iso)
    return feats


def detrend_yields(yt):
    # аномалия урожая = отклонение от технологического тренда (LOESS), в %
    out = []
    for (iso, crop), g in yt.groupby(["iso", "crop"]):
        g = g.dropna(subset=["yield"]).sort_values("year")
        if len(g) < 12:   # слишком короткий ряд - пропускаем культуру
            continue
        trend = lowess(g["yield"].values, g["year"].values.astype(float),
                       frac=0.5, return_sorted=False)
        g = g.assign(yield_trend=trend)
        g["yield_anom_pct"] = (g["yield"] - g["yield_trend"]) / g["yield_trend"] * 100.0
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    # 1. климат по всем странам
    feats = pd.concat([climate_features_for(iso) for iso in dio.COUNTRIES], ignore_index=True)
    feats.to_csv(os.path.join(PROC, "climate_features.csv"), index=False)
    print("climate_features:", feats.shape)

    # 2. ENSO, урожай (с детрендом), цены
    oni = dio.oni_features()
    yt = detrend_yields(dio.yield_table())
    price = dio.producer_price_wheat()

    # 3. склеиваем всё в одну панель
    panel = (yt.merge(feats, on=["iso", "year"], how="left")
               .merge(oni, on="year", how="left")
               .merge(price, on=["iso", "year"], how="left"))
    # лаги (прошлогодние значения) - это только прошлое, в будущее не подсматриваем
    panel = panel.sort_values(["iso", "crop", "year"])
    panel["spei6_gs_lag1"] = panel.groupby(["iso", "crop"])["spei6_gs"].shift(1)
    panel["yield_anom_lag1"] = panel.groupby(["iso", "crop"])["yield_anom_pct"].shift(1)
    panel = panel[(panel.year >= 1992) & (panel.year <= 2022)]
    panel.to_csv(os.path.join(PROC, "analysis_panel.csv"), index=False)
    print("analysis_panel:", panel.shape, "| crops:", panel.crop.nunique(),
          "| countries:", panel.iso.nunique())

    # 4. отдельно ряд по России (пшеница) - наш главный кейс
    ru = panel[(panel.iso == "RUS") & (panel.crop == "Wheat")].copy().sort_values("year")
    ru.to_csv(os.path.join(PROC, "russia_annual.csv"), index=False)
    print("russia_annual:", ru.shape, "| годы:", int(ru.year.min()), "-", int(ru.year.max()))
    print(ru[["year", "yield", "yield_anom_pct", "spei6_gs", "gs_t_z"]].round(2).tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
