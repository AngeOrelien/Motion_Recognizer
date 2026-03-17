"""
EchoSign Vision - Extracteur de keypoints (mediapipe 0.10.30+ Tasks API)
=========================================================================
mediapipe >= 0.10.30 supprime mp.solutions.* et utilise la Tasks API.
On télécharge les modèles .task au premier lancement (une seule fois).

Interface publique identique : create_holistic, mediapipe_detection,
draw_landmarks, extract_keypoints, draw_detection_status.
Vecteur de sortie : 1692 valeurs/frame (pose 132 + face 1434 + mains 126).
"""

import cv2
import numpy as np
import urllib.request
import os
from pathlib import Path

# ── Téléchargement des modèles .task ─────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "mp_models"
MODELS_DIR.mkdir(exist_ok=True)

_URLS = {
    "pose":  ("pose_landmarker_lite.task",
               "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    "face":  ("face_landmarker.task",
               "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
               "face_landmarker/float16/latest/face_landmarker.task"),
    "hands": ("hand_landmarker.task",
               "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
               "hand_landmarker/float16/latest/hand_landmarker.task"),
}

def _download_models():
    for key, (fname, url) in _URLS.items():
        dest = MODELS_DIR / fname
        if not dest.exists():
            print(f"  ⬇  Téléchargement {fname} ...")
            urllib.request.urlretrieve(url, str(dest))
            print(f"  ✓  {fname} prêt.")

_download_models()

# ── Import Tasks API ──────────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from config import (
    MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE,
)

# ── Classe _Results (émule l'ancienne interface Holistic) ─────────────────────

class _Results:
    """Conteneur unifié compatible avec le reste du code."""
    __slots__ = ("pose_landmarks", "face_landmarks",
                 "left_hand_landmarks", "right_hand_landmarks")

    def __init__(self):
        self.pose_landmarks       = None
        self.face_landmarks       = None
        self.left_hand_landmarks  = None
        self.right_hand_landmarks = None


# ── Détecteur principal ───────────────────────────────────────────────────────

class HolisticDetector:
    """Combine PoseLandmarker + FaceLandmarker + HandLandmarker (Tasks API)."""

    def __init__(self):
        BaseOptions = mp_python.BaseOptions

        # Pose
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODELS_DIR / "pose_landmarker_lite.task")),
            running_mode=VisionTaskRunningMode.IMAGE,
            min_pose_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        )
        self._pose = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        # Face
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODELS_DIR / "face_landmarker.task")),
            running_mode=VisionTaskRunningMode.IMAGE,
            min_face_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
            num_faces=1,
        )
        self._face = mp_vision.FaceLandmarker.create_from_options(face_opts)

        # Hands
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
            running_mode=VisionTaskRunningMode.IMAGE,
            min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
            num_hands=2,
        )
        self._hands = mp_vision.HandLandmarker.create_from_options(hand_opts)

    def process(self, rgb_frame) -> _Results:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        pose_res  = self._pose.detect(mp_image)
        face_res  = self._face.detect(mp_image)
        hand_res  = self._hands.detect(mp_image)

        r = _Results()

        # Pose (premier corps détecté)
        if pose_res.pose_landmarks:
            r.pose_landmarks = pose_res.pose_landmarks[0]

        # Visage (premier visage détecté)
        if face_res.face_landmarks:
            r.face_landmarks = face_res.face_landmarks[0]

        # Mains — handedness : 'Left'/'Right' du point de vue de la personne
        # Après flip horizontal de la frame, les labels sont déjà corrects.
        if hand_res.hand_landmarks:
            for lm_list, handedness_list in zip(hand_res.hand_landmarks,
                                                hand_res.handedness):
                label = handedness_list[0].category_name  # 'Left' ou 'Right'
                if label == "Left":
                    r.left_hand_landmarks  = lm_list
                else:
                    r.right_hand_landmarks = lm_list

        return r

    def close(self):
        self._pose.close()
        self._face.close()
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── API publique ──────────────────────────────────────────────────────────────

def create_holistic() -> HolisticDetector:
    return HolisticDetector()


def mediapipe_detection(frame, holistic: HolisticDetector):
    """Détection sur une frame BGR. Retourne (frame_bgr, results)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    return frame, results


def extract_keypoints(results: _Results) -> np.ndarray:
    """Vecteur 1D de 1692 valeurs (pose 132 + face 1434 + mains 126)."""
    # Pose : 33 × 4 (x, y, z, visibility)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z,
                          lm.visibility if hasattr(lm, 'visibility') else 0.0]
                         for lm in results.pose_landmarks]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # Visage : 478 × 3 (x, y, z)
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.face_landmarks]).flatten()
        # Padding si moins de 478 points
        if face.shape[0] < 478 * 3:
            face = np.concatenate([face, np.zeros(478 * 3 - face.shape[0])])
    else:
        face = np.zeros(478 * 3)

    # Mains : 21 × 3
    lh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in results.left_hand_landmarks]).flatten()
          if results.left_hand_landmarks else np.zeros(21 * 3))

    rh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in results.right_hand_landmarks]).flatten()
          if results.right_hand_landmarks else np.zeros(21 * 3))

    return np.concatenate([pose, face, lh, rh])


def draw_landmarks(image, results: _Results, draw_face: bool = False):
    """Dessine les landmarks directement sur l'image (OpenCV)."""
    h, w = image.shape[:2]

    def pt(lm):
        return int(lm.x * w), int(lm.y * h)

    # Pose — squelette simplifié
    if results.pose_landmarks:
        POSE_CONNECTIONS = [
            (11,12),(11,13),(13,15),(12,14),(14,16),
            (11,23),(12,24),(23,24),(23,25),(24,26),
            (25,27),(26,28)
        ]
        lms = results.pose_landmarks
        for a, b in POSE_CONNECTIONS:
            if a < len(lms) and b < len(lms):
                cv2.line(image, pt(lms[a]), pt(lms[b]), (80, 44, 121), 2)
        for lm in lms:
            cv2.circle(image, pt(lm), 4, (80, 22, 10), -1)

    # Mains
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    for lms, color_dot, color_line in [
        (results.left_hand_landmarks,  (121, 22, 76),  (121, 44, 250)),
        (results.right_hand_landmarks, (245, 117, 66), (245, 66, 230)),
    ]:
        if lms:
            for a, b in HAND_CONNECTIONS:
                cv2.line(image, pt(lms[a]), pt(lms[b]), color_line, 2)
            for lm in lms:
                cv2.circle(image, pt(lm), 4, color_dot, -1)

    return image


def draw_detection_status(image, results: _Results):
    """Indicateurs de détection en haut de l'image."""
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    statuses = [
        ("POSE",   results.pose_landmarks          is not None, 10),
        ("VISAGE", results.face_landmarks           is not None, 120),
        ("MAIN G", results.left_hand_landmarks      is not None, 250),
        ("MAIN D", results.right_hand_landmarks     is not None, 390),
    ]
    for label, detected, x in statuses:
        color = COLOR_GREEN if detected else COLOR_RED
        cv2.circle(image, (x + 10, 20), 7, color, -1)
        cv2.putText(image, label, (x + 22, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
    return image


def hands_detected(results: _Results) -> bool:
    return (results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None)