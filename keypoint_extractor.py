"""
EchoSign Vision - Extracteur de keypoints (MediaPipe Tasks API)
===============================================================
Compatible mediapipe 0.10.30+.

Supporte les DEUX styles utilisés dans le projet :

  Style A (data_collector.py) :
      detector.detect(frame)
      draw_landmarks(frame, detector)
      extract_keypoints(detector)

  Style B (recognizer.py) :
      image, results = mediapipe_detection(frame, holistic)
      draw_landmarks(image, results)
      extract_keypoints(results)

Les deux fonctionnent car EchoSignDetector expose directement
pose_landmarks / face_landmarks / left/right_hand_landmarks.
"""

import cv2
import numpy as np
import urllib.request
from pathlib import Path

# ── Téléchargement automatique des modèles ────────────────────────────────────
_MODELS_DIR = Path(__file__).parent / "mp_models"
_MODELS_DIR.mkdir(exist_ok=True)

_MODEL_URLS = {
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

def _ensure_models():
    for fname, url in _MODEL_URLS.items():
        dest = _MODELS_DIR / fname
        if not dest.exists():
            print(f"  Telechargement {fname} ...")
            urllib.request.urlretrieve(url, str(dest))
            print(f"  OK  {fname} pret.")

_ensure_models()

# ── Import MediaPipe Tasks ────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from config import (
    MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE,
)

# ── Connexions squelette ──────────────────────────────────────────────────────
_POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)
]
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]


# ── Classe principale ─────────────────────────────────────────────────────────

class EchoSignDetector:
    """
    Détecteur combiné Pose + Face + Hands.

    Après detect(frame), les attributs sont mis à jour :
        .pose_landmarks        -> liste 33 pts  (ou None)
        .face_landmarks        -> liste 478 pts (ou None)
        .left_hand_landmarks   -> liste 21 pts  (ou None)
        .right_hand_landmarks  -> liste 21 pts  (ou None)
    """

    def __init__(self):
        Base = mp_python.BaseOptions

        self._pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=Base(
                    model_asset_path=str(_MODELS_DIR / "pose_landmarker.task")),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_pose_detection_confidence=MP_DETECTION_CONFIDENCE,
                min_tracking_confidence=MP_TRACKING_CONFIDENCE,
            )
        )
        self._face = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=Base(
                    model_asset_path=str(_MODELS_DIR / "face_landmarker.task")),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_face_detection_confidence=MP_DETECTION_CONFIDENCE,
                min_tracking_confidence=MP_TRACKING_CONFIDENCE,
                num_faces=1,
            )
        )
        self._hands = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=Base(
                    model_asset_path=str(_MODELS_DIR / "hand_landmarker.task")),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
                min_tracking_confidence=MP_TRACKING_CONFIDENCE,
                num_hands=2,
            )
        )
        # Résultats courants
        self.pose_landmarks       = None
        self.face_landmarks       = None
        self.left_hand_landmarks  = None
        self.right_hand_landmarks = None

    def detect(self, frame_bgr):
        """
        Lance la détection sur une frame BGR.
        Met à jour tous les attributs landmarks.
        Retourne self (chaînable).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Pose
        pose_res = self._pose.detect(mp_img)
        self.pose_landmarks = (pose_res.pose_landmarks[0]
                               if pose_res.pose_landmarks else None)

        # Visage
        face_res = self._face.detect(mp_img)
        self.face_landmarks = (face_res.face_landmarks[0]
                               if face_res.face_landmarks else None)

        # Mains
        self.left_hand_landmarks  = None
        self.right_hand_landmarks = None
        hand_res = self._hands.detect(mp_img)
        if hand_res.hand_landmarks:
            for lm_list, handedness in zip(hand_res.hand_landmarks,
                                           hand_res.handedness):
                label = handedness[0].category_name   # 'Left' ou 'Right'
                if label == "Left":
                    self.left_hand_landmarks  = lm_list
                else:
                    self.right_hand_landmarks = lm_list
        return self

    def close(self):
        self._pose.close()
        self._face.close()
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Alias compatibilité recognizer.py ────────────────────────────────────────

def create_holistic() -> EchoSignDetector:
    """Crée un EchoSignDetector (utilisé par recognizer.py)."""
    return EchoSignDetector()


def mediapipe_detection(frame_bgr, detector: EchoSignDetector):
    """
    Utilisé par recognizer.py.
    Appelle detect() et retourne (frame, detector).
    Le detector expose les mêmes attributs qu'un objet results.
    """
    detector.detect(frame_bgr)
    return frame_bgr, detector


# ── Extraction des keypoints ──────────────────────────────────────────────────

def extract_keypoints(source) -> np.ndarray:
    """
    Accepte un EchoSignDetector (apres detect()) ou tout objet
    exposant pose/face/left_hand/right_hand_landmarks.
    Retourne un vecteur 1D de 1692 valeurs.
    """
    # Pose : 33 × 4 (x, y, z, visibility)
    lms = source.pose_landmarks
    if lms:
        pose = np.array([[lm.x, lm.y, lm.z,
                          getattr(lm, 'visibility', 0.0)]
                         for lm in lms]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # Visage : 478 × 3
    lms = source.face_landmarks
    if lms:
        raw = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
        target = 478 * 3
        face = np.zeros(target)
        face[:min(len(raw), target)] = raw[:target]
    else:
        face = np.zeros(478 * 3)

    # Main gauche : 21 × 3
    lms = source.left_hand_landmarks
    lh = (np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
          if lms else np.zeros(21 * 3))

    # Main droite : 21 × 3
    lms = source.right_hand_landmarks
    rh = (np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
          if lms else np.zeros(21 * 3))

    return np.concatenate([pose, face, lh, rh])   # 132+1434+63+63 = 1692


# ── Dessin ────────────────────────────────────────────────────────────────────

def draw_landmarks(image, source, draw_face: bool = False):
    """Dessine les landmarks. Accepte un EchoSignDetector ou un objet results."""
    h, w = image.shape[:2]

    def pt(lm):
        return int(lm.x * w), int(lm.y * h)

    # Pose
    lms = source.pose_landmarks
    if lms:
        for a, b in _POSE_CONNECTIONS:
            if a < len(lms) and b < len(lms):
                cv2.line(image, pt(lms[a]), pt(lms[b]), (80, 44, 121), 2)
        for lm in lms:
            cv2.circle(image, pt(lm), 4, (80, 22, 10), -1)

    # Mains
    for lms, dot_c, line_c in [
        (source.left_hand_landmarks,  (121, 22, 76),  (121, 44, 250)),
        (source.right_hand_landmarks, (245, 117, 66), (245, 66, 230)),
    ]:
        if lms:
            for a, b in _HAND_CONNECTIONS:
                cv2.line(image, pt(lms[a]), pt(lms[b]), line_c, 2)
            for lm in lms:
                cv2.circle(image, pt(lm), 4, dot_c, -1)

    return image


def draw_detection_status(image, source):
    """Indicateurs de détection en haut de l'image."""
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    statuses = [
        ("POSE",   source.pose_landmarks          is not None, 10),
        ("VISAGE", source.face_landmarks           is not None, 120),
        ("MAIN G", source.left_hand_landmarks      is not None, 250),
        ("MAIN D", source.right_hand_landmarks     is not None, 390),
    ]
    for label, detected, x in statuses:
        color = COLOR_GREEN if detected else COLOR_RED
        cv2.circle(image, (x + 10, 20), 7, color, -1)
        cv2.putText(image, label, (x + 22, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
    return image


def hands_detected(source) -> bool:
    return (source.left_hand_landmarks is not None or
            source.right_hand_landmarks is not None)