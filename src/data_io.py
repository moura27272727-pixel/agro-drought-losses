# Загрузка данных: FAOSTAT (урожай + цены), климат CCKP, индекс Эль-Ниньо (ONI).
# Заодно тут списки стран и культур, которые берём в работу.
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

# страны зернового пояса. lat - примерная широта зерновых районов (нужна для PET).
COUNTRIES = {
    "RUS": {"fao": "Russian Federation", "ru": "Россия", "lat": 53.0},
    "UKR": {"fao": "Ukraine", "ru": "Украина", "lat": 49.0},
    "KAZ": {"fao": "Kazakhstan", "ru": "Казахстан", "lat": 51.0},
    "BLR": {"fao": "Belarus", "ru": "Беларусь", "lat": 53.0},
    "MDA": {"fao": "Republic of Moldova", "ru": "Молдова", "lat": 47.0},
    "ROU": {"fao": "Romania", "ru": "Румыния", "lat": 46.0},
}
FAO2ISO = {v["fao"]: k for k, v in COUNTRIES.items()}

CROPS = ["Wheat", "Barley", "Maize (corn)", "Oats", "Rye",
         "Sunflower seed", "Potatoes", "Sugar beet"]
CROPS_RU = {
    "Wheat": "Пшеница", "Barley": "Ячмень", "Maize (corn)": "Кукуруза",
    "Oats": "Овёс", "Rye": "Рожь", "Sunflower seed": "Подсолнечник",
    "Potatoes": "Картофель", "Sugar beet": "Сахарная свёкла",
}


def load_faostat_qcl():
    # урожайность/площади/сбор по странам и культурам (длинная таблица)
    df = pd.read_csv(os.path.join(PROC, "fao_qcl_panel.csv"))
    df = df[df["Area"].isin(FAO2ISO)].copy()
    df["iso"] = df["Area"].map(FAO2ISO)
    return df[["iso", "Area", "Item", "Element", "Year", "Unit", "Value"]]


def yield_table():
    # делаем широкую таблицу: iso, year, crop -> yield, area, production
    q = load_faostat_qcl()
    q = q[q["Item"].isin(CROPS)]
    pieces = {}
    for elem, col in [("Yield", "yield"), ("Area harvested", "area"),
                      ("Production", "production")]:
        sub = q[q["Element"] == elem][["iso", "Year", "Item", "Value"]]
        sub = sub.rename(columns={"Value": col, "Year": "year", "Item": "crop"})
        pieces[col] = sub
    out = pieces["yield"]
    for col in ["area", "production"]:
        out = out.merge(pieces[col], on=["iso", "year", "crop"], how="left")
    return out


def producer_price_wheat():
    # цена пшеницы (USD/т и в нац. валюте) по странам и годам
    pp = pd.read_csv(os.path.join(PROC, "fao_pp_panel.csv"))
    pp = pp[(pp["Area"].isin(FAO2ISO)) & (pp["Item"] == "Wheat")].copy()
    if "Months" in pp.columns:           # берём только годовые значения, а не помесячные
        pp = pp[pp["Months"] == "Annual value"]
    pp["iso"] = pp["Area"].map(FAO2ISO)
    out = {}
    for elem, col in [("Producer Price (USD/tonne)", "price_usd_t"),
                      ("Producer Price (LCU/tonne)", "price_lcu_t")]:
        sub = (pp[pp["Element"] == elem][["iso", "Year", "Value"]]
               .rename(columns={"Value": col, "Year": "year"})
               .drop_duplicates(["iso", "year"]))
        out[col] = sub
    res = out["price_usd_t"].merge(out["price_lcu_t"], on=["iso", "year"], how="outer")
    return res.drop_duplicates(["iso", "year"])   # на всякий случай ещё раз дубли убрать


def load_climate(iso):
    # месячные осадки (pr, мм) и температура (tas, °C) для страны из CCKP
    if iso == "RUS":
        fpr, ftas = "cckp_pr.json", "cckp_tas.json"
    else:
        fpr, ftas = "cckp_pr_%s.json" % iso, "cckp_tas_%s.json" % iso
    pr = json.load(open(os.path.join(RAW, fpr)))["data"][iso]
    tas = json.load(open(os.path.join(RAW, ftas)))["data"][iso]
    d = pd.DataFrame({"pr": pd.Series(pr), "tas": pd.Series(tas)})
    d["year"] = d.index.str[:4].astype(int)
    d["month"] = d.index.str[5:7].astype(int)
    return d.sort_values(["year", "month"]).reset_index(drop=True)


def load_oni():
    # индекс Эль-Ниньо (ONI) по сезонам и годам из текстового файла NOAA
    path = os.path.join(RAW, "noaa_oni.txt")
    rows = []
    f = open(path)
    f.readline()   # пропускаем шапку
    for line in f:
        parts = line.split()
        if len(parts) != 4:
            continue
        seas, yr, total, anom = parts
        rows.append((seas, int(yr), float(anom)))
    f.close()
    return pd.DataFrame(rows, columns=["seas", "year", "oni"])


def oni_features():
    # годовые признаки ENSO: ONI зимы (DJF), весны (MAM), лета (JJA)
    o = load_oni()
    djf = o[o.seas == "DJF"].set_index("year")["oni"]
    mam = o[o.seas == "MAM"].set_index("year")["oni"]
    jja = o[o.seas == "JJA"].set_index("year")["oni"]
    rows = []
    for y in sorted(set(o.year)):
        rows.append({"year": y,
                     "oni_djf": djf.get(y, np.nan),
                     "oni_mam": mam.get(y, np.nan),
                     "oni_jja": jja.get(y, np.nan)})
    return pd.DataFrame(rows)
