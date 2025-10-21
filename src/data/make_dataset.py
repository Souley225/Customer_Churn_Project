"""Téléchargement du dataset Kaggle Telco Customer Churn.


- Télécharge via l'API Kaggle (requiert ~/.kaggle/kaggle.json)
- Sauvegarde sous data/raw/telco.csv
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from src.utils.paths import RAW_DIR
from src.utils.logging import logger


DATASET = "blastchar/telco-customer-churn"




def download() -> Path:
    """Télécharge le dataset et retourne le chemin du CSV principal.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Téléchargement du dataset Kaggle Telco Customer Churn…")
    subprocess.run([
    "kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--force"
    ], check=True)
    # Le zip contient WA_Fn-UseC_-Telco-Customer-Churn.csv
    zip_path = next(RAW_DIR.glob("*.zip"))
    subprocess.run(["unzip", "-o", str(zip_path), "-d", str(RAW_DIR)], check=True)
    csv_path = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    logger.info(f"CSV prêt: {csv_path}")
    return csv_path




if __name__ == "__main__":
    download()