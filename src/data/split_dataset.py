"""Découpage train/valid/test stratifié.


- Cible: Churn (Yes/No)
- Sauvegardes: data/interim/*.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.paths import INTERIM_DIR
from src.utils.io import read_csv, to_csv
from src.utils.logging import logger




def split(csv_path: str | Path, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> None:
    """Split stratifié en train/val/test.
    val_size est fraction relative à train.
    """
    df = read_csv(csv_path)
    y = df["Churn"]
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=y, random_state=random_state)


    # split validation à partir de train
    y_train = train_df["Churn"]
    train_df, val_df = train_test_split(
    train_df, test_size=val_size, stratify=y_train, random_state=random_state
    )


    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    to_csv(train_df, INTERIM_DIR / "train.csv")
    to_csv(val_df, INTERIM_DIR / "val.csv")
    to_csv(test_df, INTERIM_DIR / "test.csv")
    logger.info("Splits sauvegardés dans data/interim/")




if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()
    split(args.csv_path, args.test_size, args.val_size, args.random_state)