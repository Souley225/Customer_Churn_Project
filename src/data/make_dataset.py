"""Téléchargement du dataset Kaggle Telco Customer Churn.

- Télécharge via l'API Kaggle (requiert ~/.kaggle/kaggle.json)
- Sauvegarde sous data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
from __future__ import annotations
import subprocess
import zipfile
from pathlib import Path
from src.utils.paths import RAW_DIR
from src.utils.logging import logger


DATASET = "blastchar/telco-customer-churn"


def download() -> Path:
    """Télécharge le dataset et retourne le chemin du CSV principal.

    Raises:
        RuntimeError: Si le téléchargement échoue ou si le fichier est introuvable
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Téléchargement du dataset Kaggle Telco Customer Churn…")

    try:
        subprocess.run([
            "kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--force"
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Échec téléchargement Kaggle: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("Commande 'kaggle' introuvable. Vérifier installation kaggle CLI.") from None

    # Recherche du fichier zip téléchargé
    zip_files = list(RAW_DIR.glob("*.zip"))
    if not zip_files:
        raise RuntimeError(f"Aucun fichier ZIP trouvé dans {RAW_DIR}")
    zip_path = zip_files[0]

    # Extraction avec zipfile (compatible Windows)
    logger.info(f"Extraction de {zip_path.name}…")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Fichier ZIP corrompu: {zip_path}") from e

    csv_path = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not csv_path.exists():
        raise RuntimeError(f"CSV introuvable après extraction: {csv_path}")

    logger.info(f"CSV prêt: {csv_path}")
    return csv_path




if __name__ == "__main__":
    download()