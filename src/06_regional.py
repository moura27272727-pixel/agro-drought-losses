# Климат зернового пояса (а не всей РФ) по сеточным данным GPCC и GHCN-CAMS
# (NOAA PSL); сравнение силы связи с национальным климатом.
import json
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt

import data_io as dio
import viz
from spei import thornthwaite_pet, spei

RAW, PROC = dio.RAW, dio.PROC
# зерновой пояс: европ. РФ, Поволжье, Юж. Урал
BELT = {"lat": (45, 56), "lon": (38, 62)}
YEARS = (1992, 2019)   # где пересекаются урожай и сеточные данные

PSL = "https://downloads.psl.noaa.gov/Datasets"
GRIDS = {  # открытые сеточные файлы NOAA PSL (без авторизации)
    "ghcncams_air.nc": "%s/ghcncams/air.mon.mean.nc" % PSL,
    "gpcc_precip.nc": "%s/gpcc/full_v2020/precip.mon.total.1x1.v2020.nc" % PSL,
}


def ensure_grids():
    # загрузка сеточных файлов при отсутствии
    import subprocess
    for fn, url in GRIDS.items():
        p = os.path.join(RAW, fn)
        if not os.path.exists(p):
            print("  скачиваю", fn, "...")
            subprocess.run(["curl", "-sSL", "--max-time", "600", url, "-o", p], check=True)


def find_coord(ds, *cands):
    # имя координаты (lat/latitude)
    for c in cands:
        if c in ds.coords or c in ds.dims:
            return c
    raise KeyError("не нашёл координату из %s" % str(cands))


def belt_series(path, var, box):
    # средний по боксу месячный ряд
    ds = xr.open_dataset(path, decode_times=True)
    latn = find_coord(ds, "lat", "latitude")
    lonn = find_coord(ds, "lon", "longitude")
    da = ds[var]
    lat = ds[latn].values
    la0, la1 = box["lat"]
    lat_sl = slice(la1, la0) if lat[0] > lat[-1] else slice(la0, la1)   # широта может идти по убыванию
    da = da.sel({latn: lat_sl, lonn: slice(*box["lon"])})
    w = np.cos(np.deg2rad(da[latn]))         # косинусное взвешивание по широте
    ts = da.weighted(w).mean(dim=[latn, lonn])
    df = ts.to_dataframe(name=var).reset_index()
    t = pd.to_datetime(df["time"])
    df["year"], df["month"] = t.dt.year, t.dt.month
    return df[["year", "month", var]]


def main():
    ensure_grids()
    pr = belt_series(os.path.join(RAW, "gpcc_precip.nc"), "precip", BELT)
    ta = belt_series(os.path.join(RAW, "ghcncams_air.nc"), "air", BELT)
    if ta["air"].max() > 100:        # Кельвины -> Цельсии
        ta["air"] = ta["air"] - 273.15
    clim = (pr.merge(ta, on=["year", "month"], how="inner")
              .rename(columns={"precip": "pr", "air": "tas"})
              .dropna().sort_values(["year", "month"]).reset_index(drop=True))
    m = clim["month"].values
    clim["pet"] = thornthwaite_pet(clim["tas"].values, m, 50.0)
    clim["D"] = clim["pr"] - clim["pet"]
    clim["spei6"] = spei(clim["D"].values, m, 6)
    belt_spei = clim[clim.month == 8].set_index("year")["spei6"]
    sT = clim[clim.month.between(5, 8)].groupby("year")["tas"].mean()
    ref = sT.loc[1991:2019]
    belt_tz = (sT - ref.mean()) / ref.std()

    # национальный климат (russia_annual)
    ru = pd.read_csv(os.path.join(PROC, "russia_annual.csv")).set_index("year")
    yld = ru["yield_anom_pct"]
    nat_spei = ru["spei6_gs"]
    nat_tz = ru["gs_t_z"]

    def corr(a, b):
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        j = j[(j.index >= YEARS[0]) & (j.index <= YEARS[1])]
        r, p = stats.pearsonr(j.a, j.b)
        return round(r, 2), round(p, 3), len(j)

    res = {
        "belt_box": BELT, "years": list(YEARS),
        "national": {"spei6_vs_yield": corr(nat_spei, yld),
                     "neg_summerT_vs_yield": corr(-nat_tz, yld)},
        "grain_belt": {"spei6_vs_yield": corr(belt_spei, yld),
                       "neg_summerT_vs_yield": corr(-belt_tz, yld)},
        "spei_2010_2012": {"national": [round(nat_spei.get(2010, np.nan), 2), round(nat_spei.get(2012, np.nan), 2)],
                           "grain_belt": [round(belt_spei.get(2010, np.nan), 2), round(belt_spei.get(2012, np.nan), 2)]},
    }
    with open(os.path.join(PROC, "regional_results.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    # Рис.9
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = ["SPEI-6", "−Темп. лета"]
    nat = [res["national"]["spei6_vs_yield"][0], res["national"]["neg_summerT_vs_yield"][0]]
    belt = [res["grain_belt"]["spei6_vs_yield"][0], res["grain_belt"]["neg_summerT_vs_yield"][0]]
    x = np.arange(len(labels)); w = 0.36
    axes[0].bar(x - w / 2, nat, w, color=viz.C_NORMAL, label="Вся Россия")
    axes[0].bar(x + w / 2, belt, w, color=viz.C_DROUGHT, label="Зерновой пояс")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Корреляция с аномалией урожая")
    axes[0].set_title("Сила связи: национальный климат vs пояс")
    axes[0].legend(frameon=False, fontsize=9); axes[0].axhline(0, color="grey", lw=0.8)

    j = pd.concat([belt_spei.rename("s"), yld.rename("y")], axis=1).dropna()
    j = j[(j.index >= YEARS[0]) & (j.index <= YEARS[1])]
    axes[1].scatter(j.s, j.y, color=viz.C_DROUGHT, s=28, zorder=3)
    b = np.polyfit(j.s, j.y, 1); xs = np.linspace(j.s.min(), j.s.max(), 50)
    axes[1].plot(xs, np.polyval(b, xs), color=viz.C_NORMAL, lw=2)
    for yr in (2010, 2012):
        if yr in j.index:
            axes[1].annotate(yr, (j.loc[yr, "s"], j.loc[yr, "y"]),
                             textcoords="offset points", xytext=(6, 4), fontsize=9)
    rr = res["grain_belt"]["spei6_vs_yield"]
    axes[1].set_title("Пояс: SPEI-6 и урожай (r=%s, p=%s)" % (rr[0], rr[1]))
    axes[1].set_xlabel("SPEI-6 зернового пояса"); axes[1].set_ylabel("Аномалия урожая, %")
    axes[1].axhline(0, color="grey", lw=0.8); axes[1].axvline(0, color="grey", lw=0.8)
    viz.save(fig, "fig09_belt_vs_national")
    print("Saved fig09 and regional_results.json")


if __name__ == "__main__":
    main()
