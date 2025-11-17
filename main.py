"""Point d'entrée principal pour le déploiement Render.

Réexporte l'application FastAPI depuis src.serving.api
pour permettre à Render de la détecter automatiquement.
"""

from src.serving.api import app

__all__ = ["app"]
