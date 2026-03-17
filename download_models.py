"""
EchoSign Vision - Téléchargement des modèles MediaPipe Tasks
=============================================================
À lancer UNE SEULE FOIS avant d'utiliser le projet.

Usage :
    python download_models.py
"""

import urllib.request
import os
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "mp_models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}


def download_with_progress(url: str, dest: Path):
    """Télécharge un fichier avec barre de progression."""
    def reporthook(block, block_size, total):
        downloaded = block * block_size
        if total > 0:
            pct = min(downloaded / total * 100, 100)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            mb = downloaded / 1024 / 1024
            total_mb = total / 1024 / 1024
            print(f"\r    [{bar}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # Nouvelle ligne après la barre


def main():
    print("\n" + "=" * 60)
    print("   EchoSign Vision  —  Téléchargement des modèles MediaPipe")
    print("=" * 60)
    print(f"  Destination : {MODELS_DIR}\n")

    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  ⏭  {filename} déjà présent ({size_mb:.1f} MB), ignoré.")
            continue

        print(f"  ⬇  Téléchargement de {filename}...")
        try:
            download_with_progress(url, dest)
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  ✓  {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"\n  ❌ Erreur pour {filename} : {e}")

    print("\n" + "=" * 60)
    print("  ✅ Tous les modèles sont prêts !")
    print("  Lance maintenant : python main.py --mode collect")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
