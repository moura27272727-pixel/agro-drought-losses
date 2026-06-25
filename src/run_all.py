# запускает все шаги по порядку.
# python src/run_all.py            - всё с нуля
# python src/run_all.py --no-download  - пропустить шаг 00, если FAO-панели уже собраны
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(os.path.dirname(HERE), "data", "processed")
steps = ["00_download_data.py", "01_build_panel.py", "02_correlations.py",
         "03_panel_regression.py", "04_ml_model.py", "05_economic_loss.py",
         "06_regional.py"]
# --no-download пропускает шаг 00 ТОЛЬКО если FAO-панели уже собраны. Если
# data/processed пуст, шаг 00 всё равно выполнится: он не качает заново при наличии
# сырья в data/raw (fetch пропускает существующее), но пересоберёт
# fao_qcl_panel.csv / fao_pp_panel.csv, которые нужны шагу 01.
panels_ready = (os.path.exists(os.path.join(PROC, "fao_qcl_panel.csv"))
                and os.path.exists(os.path.join(PROC, "fao_pp_panel.csv")))
if "--no-download" in sys.argv and panels_ready:
    steps = steps[1:]

for s in steps:
    print("\n" + "=" * 60)
    print(">>>", s)
    print("=" * 60)
    subprocess.run([sys.executable, os.path.join(HERE, s)], check=True)

print("\nГотово. Графики в figures/, результаты в data/processed/.")
