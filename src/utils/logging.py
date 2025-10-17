"""Configuration de logging structuré via loguru.


Commentaires en français.
"""
from __future__ import annotations
from loguru import logger
import sys


# Configuration simple de loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")


__all__ = ["logger"]