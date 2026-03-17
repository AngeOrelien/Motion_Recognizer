"""
EchoSign Vision - Collecteur de données (MediaPipe Tasks API)
=============================================================
Usage :
    python data_collector.py
    python data_collector.py --actions bonjour merci
    python data_collector.py --resume
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path

from config import (
    DATA_DIR, ACTIONS, NB_SEQUENCES, SEQUENCE_LENGTH, COUNTDOWN,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_BLACK,
    COLOR_YELLOW, COLOR_BLUE, COLOR_PURPLE
)
from keypoint_extractor import (
    EchoSignDetector, draw_landmarks,
    extract_keypoints, draw_detection_status
)


# ── UI ────────────────────────────────────────────────────────────────────────

def draw_header(image, text, color=COLOR_BLUE):
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 40), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.putText(image, text, (w // 2 - len(text) * 7, 72),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)


def draw_progress_bar(image, current, total, label=""):
    h, w = image.shape[:2]
    bar_y, bar_h, bar_w = h - 40, 20, w - 40
    cv2.rectangle(image, (20, bar_y), (20 + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * current / max(total, 1))
    cv2.rectangle(image, (20, bar_y), (20 + fill, bar_y + bar_h), COLOR_GREEN, -1)
    cv2.putText(image, f"{label}  {current}/{total}", (25, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)


def draw_countdown(image, seconds_left):
    h, w = image.shape[:2]
    num = str(max(1, int(seconds_left) + 1))
    cv2.putText(image, num, (w // 2 - 25, h // 2 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 4, COLOR_BLACK, 10)
    cv2.putText(image, num, (w // 2 - 25, h // 2 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 4, COLOR_YELLOW, 5)
    cv2.putText(image, "PREPAREZ-VOUS", (w // 2 - 130, h // 2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)


def draw_recording(image, frame_idx):
    h, w = image.shape[:2]
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(image, (w - 30, 60), 10, COLOR_RED, -1)
        cv2.putText(image, "REC", (w - 70, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
    cv2.putText(image, f"Frame {frame_idx + 1}/{SEQUENCE_LENGTH}",
                (w - 180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)


def get_existing_sequences(action):
    action_dir = Path(DATA_DIR) / action
    if not action_dir.exists():
        return 0
    return len([f for f in action_dir.glob("*.npy") if f.stem.isdigit()])


# ── Collecte ──────────────────────────────────────────────────────────────────

def collect_action(cap, detector, action, start_seq=0):
    action_dir = Path(DATA_DIR) / action
    action_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Geste : {action.upper()}")
    print(f"  Séquences restantes : {NB_SEQUENCES - start_seq}")
    print(f"  [Q] Quitter")

    for seq_idx in range(start_seq, NB_SEQUENCES):
        sequence = []

        # Compte à rebours
        start_time = time.time()
        while time.time() - start_time < COUNTDOWN:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            detector.detect(frame)
            draw_landmarks(frame, detector)
            draw_detection_status(frame, detector)
            draw_countdown(frame, COUNTDOWN - (time.time() - start_time))
            draw_header(frame, f"Geste : {action.upper()}", COLOR_PURPLE)
            draw_progress_bar(frame, seq_idx, NB_SEQUENCES, "Séquences")
            cv2.imshow("EchoSign - Collecte", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        # Enregistrement
        for frame_idx in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            detector.detect(frame)
            draw_landmarks(frame, detector)
            draw_detection_status(frame, detector)
            draw_header(frame, f"Geste : {action.upper()}", COLOR_GREEN)
            draw_recording(frame, frame_idx)
            draw_progress_bar(frame, seq_idx, NB_SEQUENCES, "Séquences")
            cv2.imshow("EchoSign - Collecte", frame)

            sequence.append(extract_keypoints(detector))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        # Sauvegarde
        np_seq = np.array(sequence)
        np.save(str(action_dir / f"{seq_idx}.npy"), np_seq)
        print(f"  ✓ Séquence {seq_idx + 1:02d}/{NB_SEQUENCES}  {np_seq.shape}")

    return True


def run_collection(actions_list=None, resume=False):
    actions = actions_list or ACTIONS

    print("\n" + "=" * 60)
    print("   EchoSign Vision  —  Collecte de Données")
    print("=" * 60)

    input("  Appuie sur [ENTRÉE] pour commencer...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Impossible d'accéder à la webcam !")
        return

    with EchoSignDetector() as detector:
        for action in actions:
            start_seq = 0
            if resume:
                start_seq = min(get_existing_sequences(action), NB_SEQUENCES)
                if start_seq >= NB_SEQUENCES:
                    print(f"  ⏭  {action} déjà complet.")
                    continue

            ok = collect_action(cap, detector, action, start_seq)
            if not ok:
                print("  ⛔ Collecte interrompue.")
                break
            print(f"  ✅ '{action}' terminé !")

    cap.release()
    cv2.destroyAllWindows()

    print("\n  RÉSUMÉ :")
    for a in ACTIONS:
        n = get_existing_sequences(a)
        bar = "█" * n + "░" * (NB_SEQUENCES - n)
        print(f"  {a:20s} [{bar}] {n}/{NB_SEQUENCES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_collection(actions_list=args.actions, resume=args.resume)
