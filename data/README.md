# Источники данных

Все данные публичные, открытого доступа, **без выдуманных значений**.
Дата выгрузки: **25 июня 2026 г.** Скрипт загрузки: [`src/00_download_data.py`](../src/00_download_data.py).

| Данные | Источник | URL | Лицензия |
|---|---|---|---|
| Урожайность, площади, валовой сбор (зерновые и др.), 1992–2024 | FAOSTAT, домен *Production: Crops and livestock products* (QCL) | https://bulks-faostat.fao.org/production/ | CC BY 4.0 |
| Цены производителей (USD/т, нац. валюта/т), 1992–2025 | FAOSTAT, домен *Prices: Producer Prices* (PP) | https://bulks-faostat.fao.org/production/ | CC BY 4.0 |
| Месячные осадки и температура по странам, 1901–2022 (CRU TS 4.07) | World Bank Climate Change Knowledge Portal (CCKP) | https://cckpapi.worldbank.org/ | CC BY 4.0 |
| Индекс ENSO: Oceanic Niño Index (ONI), 1950–2026 | NOAA Climate Prediction Center | https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt | Public domain (U.S. Gov.) |
| Урожайность зерновых, сверка чтения FAOSTAT | World Bank Open Data API (`AG.YLD.CREL.KG`) | https://api.worldbank.org/v2/country/RUS/indicator/AG.YLD.CREL.KG | CC BY 4.0 |
| Урожайность пшеницы, сверка чтения FAOSTAT | Our World in Data (по данным ФАО) | https://ourworldindata.org/grapher/wheat-yields | CC BY 4.0 |

> World Bank и OWID производны от данных ФАО, поэтому подтверждают лишь корректность
> чтения FAOSTAT, но не являются независимой валидацией. Полностью независимая проверка
> потребовала бы национальной статистики (Росстат) и вынесена в направления развития.
| Сеточные осадки (1°, для климата зернового пояса) | GPCC v2020, NOAA PSL | https://psl.noaa.gov/data/gridded/data.gpcc.html | открыто |
| Сеточная температура (0.5°, для климата зернового пояса) | GHCN-CAMS, NOAA PSL | https://psl.noaa.gov/data/gridded/data.ghcncams.html | открыто |

> Сеточные файлы (`ghcncams_air.nc`, `gpcc_precip.nc`) крупные, в репозиторий не входят
> и автоматически скачиваются скриптом `src/06_regional.py` (региональный анализ, 
> климат зернового пояса вместо усреднения по всей стране).

## Страны панели
Черноморский / постсоветский зерновой пояс: **Россия (главный кейс)**, Украина,
Казахстан, Беларусь, Молдова, Румыния.

## Структура каталога
- `raw/`: первичные выгрузки (в репозиторий не входят, см. `.gitignore`;
  восстанавливаются скриптом `00_download_data.py`).
- `processed/`: производные таблицы:
  - `analysis_panel.csv`, панель «страна × год × культура» с климатическими
    признаками и аномалиями урожайности;
  - `climate_features.csv`, климатические признаки (SPEI/SPI, аномалии);
  - `russia_annual.csv`, годовой ряд по России (пшеница);
  - `results_correlations.csv`, `panel_results.json`, `ml_results.json`,
    `economic_loss_*.{csv,json}`, итоговые результаты.

## Замечание про EM-DAT
База бедствий **EM-DAT** (https://www.emdat.be) содержит отчётный ущерб от засух,
но требует бесплатной регистрации и не допускает автоматическую выгрузку, поэтому
в пайплайн не включена. Денежные убытки в работе оцениваются независимо, через
недобор урожая относительно тренда и цены производителя FAOSTAT, а опубликованные
оценки засух 2010/2012 гг. используются для сопоставления (см. отчёт, раздел обзора).
