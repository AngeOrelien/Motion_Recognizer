"""
EchoSign Vision - Reconnaissance en temps réel
===============================================
Charge le modèle entraîné et prédit les gestes en direct
depuis la webcam, avec un historique visuel et des barres de confiance.

Usage :
    python recognizer.py
"""

import os
import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from config import (
    MODEL_PATH, LABEL_PATH, SEQUENCE_LENGTH,
    CONFIDENCE_THRESHOLD,
    COLOR_GREEN, COLOR_YELLOW, COLOR_WHITE, COLOR_PURPLE
)
from keypoint_extractor import (
    create_holistic, mediapipe_detection,
    draw_landmarks, extract_keypoints,
    draw_detection_status
)


# ── Couleurs par classe (générées automatiquement) ────────────────────────────
CLASS_COLORS = [
    (52, 152, 219),   # bleu
    (46, 204, 113),   # vert
    (231, 76, 60),    # rouge
    (241, 196, 15),   # jaune
    (155, 89, 182),   # violet
    (26, 188, 156),   # turquoise
    (230, 126, 34),   # orange
    (236, 240, 241),  # blanc cassé
]


# ── UI ────────────────────────────────────────────────────────────────────────

def draw_prediction_box(image, prediction: str, confidence: float):
    """Affiche le résultat principal en grand."""
    h, w = image.shape[:2]
    box_h = 80

    # Fond
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - box_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    # Couleur selon confiance
    if confidence >= 0.90:
        color = COLOR_GREEN
    elif confidence >= CONFIDENCE_THRESHOLD:
        color = COLOR_YELLOW
    else:
        color = (100, 100, 100)

    # Texte
    display = prediction.replace("_", " ").upper()
    cv2.putText(image, display,
                (20, h - box_h + 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 3)
    cv2.putText(image, f"{confidence * 100:.1f}%",
                (w - 120, h - box_h + 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
    return image


def draw_confidence_bars(image, probs: np.ndarray, labels: list):
    """Barres de confiance verticales pour toutes les classes."""
    h, w = image.shape[:2]
    n = len(labels)
    bar_zone_w = 200
    bar_zone_h = h - 160
    bar_w = max(15, (bar_zone_w - 10) // n - 4)
    start_x = w - bar_zone_w - 10
    start_y = 50

    # Fond
    overlay = image.copy()
    cv2.rectangle(overlay, (start_x - 5, start_y),
                  (w - 5, start_y + bar_zone_h + 30), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    for i, (label, prob) in enumerate(zip(labels, probs)):
        x = start_x + i * (bar_w + 4)
        color = CLASS_COLORS[i % len(CLASS_COLORS)]

        # Barre vide
        cv2.rectangle(image,
                      (x, start_y),
                      (x + bar_w, start_y + bar_zone_h),
                      (60, 60, 60), -1)
        # Remplissage
        fill = int(bar_zone_h * prob)
        cv2.rectangle(image,
                      (x, start_y + bar_zone_h - fill),
                      (x + bar_w, start_y + bar_zone_h),
                      color, -1)

        # Label
        short = label[:4].upper()
        cv2.putText(image, short,
                    (x, start_y + bar_zone_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1)
    return image


def draw_history(image, history: list, y_offset: int = 95):
    """Affiche l'historique des 5 dernières prédictions."""
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y_offset), (300, y_offset + 30 * 5 + 10),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    cv2.putText(image, "Historique :", (10, y_offset + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    for i, (word, conf) in enumerate(reversed(history[-5:])):
        alpha = 1.0 - i * 0.15
        gray = int(255 * alpha)
        color = (gray, gray, gray)
        text = f"  {word.replace('_', ' '):16s} {conf*100:4.1f}%"
        cv2.putText(image, text,
                    (10, y_offset + 15 + (i + 1) * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return image


def draw_fps(image, fps: float):
    h, w = image.shape[:2]
    cv2.putText(image, f"FPS: {fps:.1f}",
                (w - 110, h - 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return image


# ── Reconnaissance ────────────────────────────────────────────────────────────

def run_recognition():
    """Boucle principale de reconnaissance en temps réel."""

    # Vérification du modèle
    if not Path(MODEL_PATH).exists():
        print("❌ Modèle introuvable. Lance d'abord train_model.py !")
        return
    if not Path(LABEL_PATH).exists():
        print("❌ Fichier de labels introuvable !")
        return

    print("\n" + "=" * 60)
    print("   EchoSign Vision  —  Reconnaissance en Temps Réel")
    print("=" * 60)

    # Chargement
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABEL_PATH, allow_pickle=True).tolist()
    print(f"  ✅ Modèle chargé   : {MODEL_PATH}")
    print(f"  ✅ Classes ({len(labels)}) : {labels}")
    print(f"  Seuil confiance  : {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print("\n  [Q] Quitter  |  [R] Réinitialiser l'historique")
    print("=" * 60)

    # État
    sequence     = deque(maxlen=SEQUENCE_LENGTH)   # Buffer de frames
    pred_history = []                               # Historique des prédictions
    current_pred = ""
    current_conf = 0.0
    smooth_probs = np.zeros(len(labels))            # Lissage temporel
    smooth_alpha = 0.4

    # FPS
    fps_times = deque(maxlen=30)
    last_time = time.time()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Impossible d'accéder à la webcam !")
        return

    with create_holistic() as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            draw_detection_status(image, results)

            # ── FPS ───────────────────────────────────────────────────────
            now = time.time()
            fps_times.append(now - last_time)
            last_time = now
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0

            # ── Extraction + buffer ───────────────────────────────────────
            kp = extract_keypoints(results)
            sequence.append(kp)

            # ── Prédiction dès que le buffer est plein ────────────────────
            if len(sequence) == SEQUENCE_LENGTH:
                input_seq = np.expand_dims(np.array(sequence), axis=0)
                probs = model.predict(input_seq, verbose=0)[0]

                # Lissage exponentiel
                smooth_probs = smooth_alpha * probs + (1 - smooth_alpha) * smooth_probs
                pred_idx  = np.argmax(smooth_probs)
                pred_conf = smooth_probs[pred_idx]

                if pred_conf >= CONFIDENCE_THRESHOLD:
                    new_pred = labels[pred_idx]
                    if new_pred != current_pred or pred_conf > current_conf + 0.05:
                        current_pred = new_pred
                        current_conf = pred_conf
                        # Ajouter à l'historique si changement significatif
                        if (not pred_history or
                                pred_history[-1][0] != current_pred):
                            pred_history.append((current_pred, current_conf))

                # ── Barres de confiance ───────────────────────────────────
                draw_confidence_bars(image, smooth_probs, labels)

            # ── Affichage principal ───────────────────────────────────────
            if current_pred:
                draw_prediction_box(image, current_pred, current_conf)
            if pred_history:
                draw_history(image, pred_history)

            draw_fps(image, fps)

            # En-tête
            h_img, w_img = image.shape[:2]
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 40), (w_img, 90), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
            cv2.putText(image, "ECHOSIGN VISION  —  Reconnaissance Active",
                        (w_img // 2 - 220, 72),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_PURPLE, 2)

            # Instruction si buffer pas encore plein
            if len(sequence) < SEQUENCE_LENGTH:
                pct = int(len(sequence) / SEQUENCE_LENGTH * 100)
                cv2.putText(image, f"Initialisation... {pct}%",
                            (20, h_img // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2)

            cv2.imshow("EchoSign - Reconnaissance", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                sequence.clear()
                pred_history.clear()
                current_pred = ""
                current_conf = 0.0
                smooth_probs = np.zeros(len(labels))
                print("  🔄 Historique réinitialisé")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Session terminée.")


if __name__ == "__main__":
    run_recognition()
