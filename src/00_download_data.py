# Качаем все исходные данные (всё открытое, без ключей).
# FAOSTAT (урожай + цены), климат CCKP, индекс ONI, плюс пара источников для проверки.
# python src/00_download_data.py            - не качает то что уже есть
# python src/00_download_data.py --force    - перекачать всё
import os
import sys
import subprocess
import zipfile
import glob
import pandas as pd

import data_io as dio

RAW, PROC = dio.RAW, dio.PROC
os.makedirs(RAW, exist_ok=True)
os.makedirs(PROC, exist_ok=True)
FORCE = "--force" in sys.argv

FAO = "https://bulks-faostat.fao.org/production"
CCKP = ("https://cckpapi.worldbank.org/cckp/v1/"
        "cru-x0.5_timeseries_{var}_timeseries_monthly_1901-2022_mean_historical_"
        "cru_ts4.07_mean/{iso}?_format=json")


def fetch(url, dest, timeout=600):
    # скачиваем через curl (на маке работает стабильнее python-овского ssl)
    if os.path.exists(dest) and not FORCE:
        print("  skip (есть):", os.path.basename(dest))
        return
    print("  GET", url)
    subprocess.run(["curl", "-sSL", "--max-time", str(timeout), url, "-o", dest], check=True)


def filt(pattern, out, items=None):
    # из большого csv FAOSTAT оставляем только наши страны (и культуры)
    files = [x for x in glob.glob(os.path.join(RAW, pattern))
             if all(k not in x for k in ["AreaCodes", "Flags", "ItemCodes", "Elements", "NOFLAG"])]
    f = files[0]
    countries = [v["fao"] for v in dio.COUNTRIES.values()]
    chunks = []
    for ch in pd.read_csv(f, encoding="utf-8-sig", chunksize=300000, low_memory=False):
        sub = ch[ch["Area"].isin(countries)]
        if items:
            sub = sub[sub["Item"].isin(items)]
        chunks.append(sub)
    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(out, index=False)
    print("  ->", os.path.basename(out), df.shape)


def main():
    # FAOSTAT (большие zip-архивы)
    print("FAOSTAT bulk:")
    qcl_zip = os.path.join(RAW, "QCL.zip")
    pp_zip = os.path.join(RAW, "PP.zip")
    fetch(FAO + "/Production_Crops_Livestock_E_All_Data_(Normalized).zip", qcl_zip)
    fetch(FAO + "/Prices_E_All_Data_(Normalized).zip", pp_zip)
    for z in (qcl_zip, pp_zip):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)
    filt("Production_Crops_Livestock_E_All_Data*.csv", os.path.join(PROC, "fao_qcl_panel.csv"), dio.CROPS)
    filt("Prices_E_All_Data*.csv", os.path.join(PROC, "fao_pp_panel.csv"))

    # климат CCKP по каждой стране
    print("CCKP климат:")
    for iso in dio.COUNTRIES:
        for var in ("pr", "tas"):
            suffix = "" if iso == "RUS" else "_" + iso   # для России файлы исторически без суффикса
            name = "cckp_%s%s.json" % (var, suffix)
            fetch(CCKP.format(var=var, iso=iso), os.path.join(RAW, name), timeout=90)

    # индекс Эль-Ниньо
    print("NOAA ONI:")
    fetch("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
          os.path.join(RAW, "noaa_oni.txt"), timeout=90)

    # для перекрёстной проверки урожайности
    print("Кросс-проверка (World Bank API, OWID):")
    fetch("https://api.worldbank.org/v2/country/RUS/indicator/AG.YLD.CREL.KG?"
          "format=json&per_page=100&date=1992:2023",
          os.path.join(RAW, "wb_cereal_yield.json"), timeout=90)
    fetch("https://ourworldindata.org/grapher/wheat-yields.csv?csvType=full",
          os.path.join(RAW, "owid_wheat.csv"), timeout=90)
    print("Готово. Данные в data/raw и data/processed.")


if __name__ == "__main__":
    main()
