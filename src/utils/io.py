"""Fonctions I/O (lecture/écriture) centralisées.


Commentaires en français.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd




def read_csv(path: Path | str) -> pd.DataFrame:
    """Lit un CSV avec dtype inféré prudemment.
    Args:
    path: chemin du fichier CSV
    Returns:
    DataFrame
    """
    return pd.read_csv(path)




def to_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Écrit un DataFrame en CSV en créant les dossiers au besoin.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)