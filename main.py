"""
EchoSign Vision - Point d'entrée principal
==========================================
Usage :
    python main.py --mode collect
    python main.py --mode collect --resume
    python main.py --mode train
    python main.py --mode train --epochs 300
    python main.py --mode recognize
    python main.py --mode status
"""

import argparse
import sys
import os
from pathlib import Path


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        ███████╗ ██████╗██╗  ██╗ ██████╗                      ║
║        ██╔════╝██╔════╝██║  ██║██╔═══██╗                     ║
║        █████╗  ██║     ███████║██║   ██║                     ║
║        ██╔══╝  ██║     ██╔══██║██║   ██║                     ║
║        ███████╗╚██████╗██║  ██║╚██████╔╝                     ║
║        ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝                      ║
║                                                              ║
║         SIGN VISION  —  Reconnaissance Langue des Signes     ║
║                     MediaPipe + LSTM                         ║
╚══════════════════════════════════════════════════════════════╝
    """)


def cmd_status():
    """Affiche l'état du dataset et du modèle."""
    from config import DATA_DIR, MODEL_PATH, LABEL_PATH, ACTIONS, NB_SEQUENCES, SEQUENCE_LENGTH
    import numpy as np

    print_banner()
    print("  ── ÉTAT DU PROJET ──────────────────────────────────────\n")

    # Dataset
    print("  📁 Dataset :")
    total_seq = 0
    for action in ACTIONS:
        action_dir = Path(DATA_DIR) / action
        count = len(list(action_dir.glob("*.npy"))) if action_dir.exists() else 0
        total_seq += count
        pct  = count / NB_SEQUENCES * 100
        bar  = "█" * count + "░" * (NB_SEQUENCES - count)
        mark = "✓" if count >= NB_SEQUENCES else "·"
        print(f"    {mark} {action:20s} [{bar}] {count:2d}/{NB_SEQUENCES}")
    print(f"\n    Total : {total_seq} séquences  "
          f"({total_seq * SEQUENCE_LENGTH} frames)\n")

    # Modèle
    print("  🧠 Modèle :")
    if Path(MODEL_PATH).exists():
        size = Path(MODEL_PATH).stat().st_size / 1024 / 1024
        print(f"    ✓ Modèle trouvé   : {MODEL_PATH}  ({size:.1f} MB)")
    else:
        print(f"    ✗ Aucun modèle    : lance --mode train")

    if Path(LABEL_PATH).exists():
        labels = np.load(LABEL_PATH, allow_pickle=True)
        print(f"    ✓ Labels ({len(labels)})     : {list(labels)}")
    else:
        print(f"    ✗ Aucun fichier de labels")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="EchoSign Vision — Reconnaissance de langue des signes",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["collect", "train", "recognize", "status"],
        default="status",
        help=(
            "collect   : Enregistrer les gestes via webcam\n"
            "train     : Entraîner le modèle LSTM\n"
            "recognize : Reconnaissance en temps réel\n"
            "status    : Voir l'état du dataset et du modèle"
        )
    )
    parser.add_argument("--resume",  action="store_true",
                        help="(collect) Reprendre la collecte")
    parser.add_argument("--actions", nargs="+", default=None,
                        help="(collect) Gestes à enregistrer")
    parser.add_argument("--epochs",  type=int, default=None,
                        help="(train) Nombre d'époques")
    parser.add_argument("--batch",   type=int, default=None,
                        help="(train) Taille du batch")

    args = parser.parse_args()
    print_banner()

    if args.mode == "status":
        cmd_status()

    elif args.mode == "collect":
        from data_collector import run_collection
        run_collection(actions_list=args.actions, resume=args.resume)

    elif args.mode == "train":
        from train_model import train
        from config import EPOCHS, BATCH_SIZE
        train(
            epochs=args.epochs    or EPOCHS,
            batch_size=args.batch or BATCH_SIZE
        )

    elif args.mode == "recognize":
        from recognizer import run_recognition
        run_recognition()


if __name__ == "__main__":
    main()
