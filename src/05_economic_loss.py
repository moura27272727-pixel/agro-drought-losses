# Считаем деньги: сколько Россия потеряла на недоборе зерна в засухи.
# Недобор = ожидаемый урожай минус фактический (только когда он положительный).
# Ожидаемый берём двумя способами (тренд LOESS и среднее за 3 прошлых года) и в
# двух ценах (текущих и постоянных) - чтобы показать что вывод устойчив.
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_io as dio
import viz

PROC = dio.PROC
GRAINS = ["Wheat", "Barley", "Maize (corn)", "Oats", "Rye"]


def grain_prices():
    # цены производителя по зерновым (USD/т, годовые) для России
    pp = pd.read_csv(os.path.join(PROC, "fao_pp_panel.csv"))
    if "Months" in pp.columns:
        pp = pp[pp["Months"] == "Annual value"]
    pp = pp[(pp.Area == "Russian Federation") &
            (pp.Element == "Producer Price (USD/tonne)") &
            (pp.Item.isin(GRAINS))]
    return (pp[["Item", "Year", "Value"]]
            .rename(columns={"Item": "crop", "Year": "year", "Value": "price_usd_t"})
            .drop_duplicates(["crop", "year"]))


def main():
    panel = pd.read_csv(os.path.join(PROC, "analysis_panel.csv"))
    ru = panel[(panel.iso == "RUS") & (panel.crop.isin(GRAINS))].copy().sort_values(["crop", "year"])
    prices = grain_prices().rename(columns={"price_usd_t": "crop_price"})

    # постоянная цена = среднее за 2014-2020 по культуре. В панели price_usd_t -
    # это цена пшеницы, используем её как запасную если по культуре цены нет.
    const = (prices[prices.year.between(2014, 2020)].groupby("crop")["crop_price"]
             .mean().rename("price_const").reset_index())
    ru = (ru.merge(prices, on=["crop", "year"], how="left")
            .merge(const, on="crop", how="left"))
    ru["price_cur"] = ru["crop_price"].fillna(ru["price_usd_t"])
    ru["price_const"] = ru["price_const"].fillna(const[const.crop == "Wheat"]["price_const"].iloc[0])

    # второй контрфакт: среднее за 3 предыдущих года
    ru["prev3"] = ru.groupby("crop")["yield"].transform(lambda s: s.shift(1).rolling(3).mean())

    def losses(expected_col, price_col):
        short = (ru[expected_col] - ru["yield"]).clip(lower=0) / 1000.0   # т/га недобора
        tonnes = short * ru["area"]
        value = tonnes * ru[price_col]
        exp_val = (ru[expected_col] / 1000.0) * ru["area"] * ru[price_col]   # ожидаемая стоимость
        tmp = pd.DataFrame({"year": ru.year, "t": tonnes, "v": value, "e": exp_val})
        g = tmp.groupby("year").sum()
        g["share"] = g["v"] / g["e"] * 100
        return g

    loess_cur = losses("yield_trend", "price_cur")    # основной вариант (консервативный)
    loess_con = losses("yield_trend", "price_const")
    prev3_cur = losses("prev3", "price_cur")
    prev3_con = losses("prev3", "price_const")

    out = loess_cur.rename(columns={"t": "tonnes", "v": "value_usd", "e": "exp_usd",
                                    "share": "loss_share_pct"}).reset_index()
    out["value_lost_bln"] = out["value_usd"] / 1e9
    out["mt_lost"] = out["tonnes"] / 1e6
    out.to_csv(os.path.join(PROC, "economic_loss_russia.csv"), index=False)

    def top(g, n=6):
        gg = g.sort_values("v", ascending=False).head(n)
        return [{"year": int(y), "mt": round(r.t / 1e6, 1), "bln_usd": round(r.v / 1e9, 2),
                 "share_pct": round(r.share, 1)} for y, r in gg.iterrows()]

    def years(g, yy):
        return {int(y): round(g.loc[y, "v"] / 1e9, 2) for y in yy if y in g.index}

    summary = {
        "headline_method": "LOESS-тренд, текущие цены (консервативно)",
        "top_years_headline": top(loess_cur),
        "robustness_bln_usd": {
            "2010": {"loess_cur": years(loess_cur, [2010])[2010], "loess_const": years(loess_con, [2010])[2010],
                     "prev3_cur": years(prev3_cur, [2010])[2010], "prev3_const": years(prev3_con, [2010])[2010]},
            "2012": {"loess_cur": years(loess_cur, [2012])[2012], "loess_const": years(loess_con, [2012])[2012],
                     "prev3_cur": years(prev3_cur, [2012])[2012], "prev3_const": years(prev3_con, [2012])[2012]},
            "2021": {"loess_cur": years(loess_cur, [2021])[2021], "loess_const": years(loess_con, [2021])[2021],
                     "prev3_cur": years(prev3_cur, [2021])[2021], "prev3_const": years(prev3_con, [2021])[2021]},
        },
        "cumulative_2000_2022_bln_usd": {
            "loess_cur": round(loess_cur.loc[2000:2022, "v"].sum() / 1e9, 1),
            "prev3_cur": round(prev3_cur.loc[2000:2022, "v"].sum() / 1e9, 1),
        },
    }
    with open(os.path.join(PROC, "economic_loss_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Рис.8 потери по годам: столбцы - тренд LOESS, линия - второй контрфакт
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(out.year, out.value_lost_bln, color=viz.C_DROUGHT, alpha=0.85, label="LOESS-тренд (консерв.)")
    p3 = (prev3_cur["v"] / 1e9).reindex(out.year).values
    ax.plot(out.year, p3, "o-", color=viz.C_NORMAL, ms=3, lw=1.4, label="Контрфакт: ср. 3 пред. года")
    for _, r in out.iterrows():
        if r.value_lost_bln > 2:
            ax.annotate("%d" % int(r.year), (r.year, r.value_lost_bln),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_title("Оценка прямых потерь сбора зерновых от недобора урожая, Россия")
    ax.set_xlabel("Год"); ax.set_ylabel("Потери, млрд долл. США")
    ax.legend(frameon=False, fontsize=9)
    viz.save(fig, "fig08_economic_loss")
    print("Saved economic loss outputs")


if __name__ == "__main__":
    main()
