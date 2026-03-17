"""
EchoSign Vision - Configuration centrale (MediaPipe Tasks API)
==============================================================
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
#  CHEMINS
# ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "dataset"
MODEL_DIR  = BASE_DIR / "models"
LOG_DIR    = BASE_DIR / "logs"
MP_MODELS  = BASE_DIR / "mp_models"   # Modèles MediaPipe Tasks (.task)

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, MP_MODELS]:
    d.mkdir(exist_ok=True)

MODEL_PATH = str(MODEL_DIR / "echosign_lstm.h5")
LABEL_PATH = str(MODEL_DIR / "labels.npy")

# Chemins des modèles MediaPipe Tasks
HAND_MODEL = str(MP_MODELS / "hand_landmarker.task")
POSE_MODEL = str(MP_MODELS / "pose_landmarker.task")
FACE_MODEL = str(MP_MODELS / "face_landmarker.task")

# ──────────────────────────────────────────────
#  GESTES À RECONNAÎTRE
# ──────────────────────────────────────────────
ACTIONS = [
    "bonjour",
    "merci",
    "oui",
    "non",
    "aide",
    "comment_ca_va",
    "au_revoir",
]

# ──────────────────────────────────────────────
#  PARAMÈTRES DE COLLECTE
# ──────────────────────────────────────────────
NB_SEQUENCES    = 40
SEQUENCE_LENGTH = 30
COUNTDOWN       = 2

# ──────────────────────────────────────────────
#  PARAMÈTRES MEDIAPIPE TASKS
# ──────────────────────────────────────────────
MP_DETECTION_CONFIDENCE = 0.7
MP_TRACKING_CONFIDENCE  = 0.5

# ──────────────────────────────────────────────
#  PARAMÈTRES DU MODÈLE LSTM
# ──────────────────────────────────────────────
EPOCHS           = 200
BATCH_SIZE       = 32
VALIDATION_SPLIT = 0.15
LEARNING_RATE    = 0.0005
CONFIDENCE_THRESHOLD = 0.80

# ──────────────────────────────────────────────
#  TAILLE DES KEYPOINTS (nouvelle API Tasks)
#  pose(33×4) + face(478×3) + main_g(21×3) + main_d(21×3)
#  Note : 478 points visage (vs 468 dans l'ancienne API)
# ──────────────────────────────────────────────
POSE_KP      = 33  * 4   # 132
FACE_KP      = 478 * 3   # 1434
HAND_KP      = 21  * 3   # 63 (× 2)
KEYPOINT_SIZE = POSE_KP + FACE_KP + HAND_KP * 2  # 1692

# ──────────────────────────────────────────────
#  COULEURS BGR (OpenCV)
# ──────────────────────────────────────────────
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_BLUE   = (255, 100, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_PURPLE = (200, 50, 200)
