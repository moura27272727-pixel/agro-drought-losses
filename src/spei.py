# Индексы засухи SPI и SPEI + испаряемость PET для них.
# SPI - по осадкам через гамма-распределение (McKee 1993).
# SPEI - по балансу P-PET через лог-логистику (Vicente-Serrano 2010).
# PET считаю по Торнтвейту. Своими руками, без готовых библиотек по засухе,
# чтобы было видно что внутри (и чтобы препод не спросил "а это что за пакет").

import numpy as np
from scipy.stats import gamma, norm
from scipy.special import gamma as gamma_fn


def daylight(lat_deg, month):
    # сколько часов длится день (широта lat, середина месяца) - нужно для PET
    lat = np.radians(lat_deg)
    doy = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])[month - 1]
    decl = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)   # склонение солнца
    cos_ws = -np.tan(lat) * np.tan(decl)
    cos_ws = np.clip(cos_ws, -1.0, 1.0)                   # чтобы arccos не упал
    ws = np.arccos(cos_ws)
    return 24.0 / np.pi * ws


def thornthwaite_pet(temp_c, months, lat_deg):
    # потенциальная испаряемость по Торнтвейту, мм/мес
    temp_c = np.asarray(temp_c, dtype=float)
    months = np.asarray(months, dtype=int)
    # тепловой индекс I по средним температурам каждого месяца
    monthly_mean = np.array([temp_c[months == m].mean() for m in range(1, 13)])
    i_m = np.where(monthly_mean > 0, (np.maximum(monthly_mean, 0) / 5.0) ** 1.514, 0.0)
    I = i_m.sum()
    a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239
    ndays = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    pet = np.zeros_like(temp_c)
    for k, (t, m) in enumerate(zip(temp_c, months)):
        if t <= 0 or I == 0:
            pet[k] = 0.0   # мороз - испарения считай что нет
            continue
        N = daylight(lat_deg, m)
        corr = (N / 12.0) * (ndays[m - 1] / 30.0)   # поправка на день и число дней
        pet[k] = 16.0 * corr * (10.0 * t / I) ** a
    return pet


def roll_sum(x, scale):
    # скользящая сумма за scale месяцев (в начале будут NaN)
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[scale - 1:] = c[scale:] - c[:-scale]
    return out


def pwm(sample):
    # взвешенные по вероятности моменты w0,w1,w2 (для лог-логистики)
    x = np.sort(np.asarray(sample, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    F = (i - 0.35) / n
    w0 = x.mean()
    w1 = np.sum((1 - F) ** 1 * x) / n
    w2 = np.sum((1 - F) ** 2 * x) / n
    return w0, w1, w2


def loglog_cdf(values, sample):
    # CDF лог-логистического распределения, параметры по sample
    w0, w1, w2 = pwm(sample)
    denom = (6 * w1 - w0 - 6 * w2)
    if denom == 0:
        return np.full_like(values, np.nan, dtype=float)
    beta = (2 * w1 - w0) / denom
    g1 = gamma_fn(1 + 1 / beta)
    g2 = gamma_fn(1 - 1 / beta)
    alpha = (w0 - 2 * w1) * beta / (g1 * g2)
    gam = w0 - alpha * g1 * g2
    z = (values - gam) / alpha
    with np.errstate(invalid="ignore", divide="ignore"):   # тут бывают деления, глушим варнинги
        cdf = np.where(z > 0, 1.0 / (1.0 + z ** (-beta)), np.nan)
    return cdf


def gamma_cdf(values, sample):
    # CDF гаммы методом Тома (нулевые осадки обрабатываем отдельно)
    sample = np.asarray(sample, dtype=float)
    pos = sample[sample > 0]
    n = len(sample)
    q0 = (n - len(pos)) / n if n else 0.0   # доля нулей
    if len(pos) < 4:
        return np.full_like(values, np.nan, dtype=float)
    A = np.log(pos.mean()) - np.log(pos).mean()
    shape = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A)
    scale = pos.mean() / shape
    g = gamma.cdf(values, a=shape, scale=scale)
    return q0 + (1 - q0) * g


def standardize(acc, months, ref_mask, kind):
    # переводим накопленные суммы в N(0,1), калибруем отдельно по каждому месяцу
    out = np.full_like(acc, np.nan, dtype=float)
    for m in range(1, 13):
        sel = months == m
        sample = acc[sel & ref_mask]
        sample = sample[~np.isnan(sample)]
        vals = acc[sel]
        if len(sample) < 8:   # мало данных - пропускаем
            continue
        if kind == "spei":
            cdf = loglog_cdf(vals, sample)
        else:
            cdf = gamma_cdf(vals, sample)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        out[sel] = norm.ppf(cdf)
    return np.clip(out, -3.0, 3.0)   # обрезаем хвосты, как обычно для индексов засухи


def spi(precip, months, scale, ref_mask=None):
    # SPI за scale месяцев
    acc = roll_sum(precip, scale)
    if ref_mask is None:
        ref_mask = ~np.isnan(acc)
    return standardize(acc, months, ref_mask, "spi")


def spei(water_balance, months, scale, ref_mask=None):
    # SPEI за scale месяцев, на вход подаём баланс P-PET
    acc = roll_sum(water_balance, scale)
    if ref_mask is None:
        ref_mask = ~np.isnan(acc)
    return standardize(acc, months, ref_mask, "spei")
