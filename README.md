# Motion_Recognizer
# EchoSign Vision 🤟
### Reconnaissance de Langue des Signes — MediaPipe + LSTM Bidirectionnel

---

## 🏗️ Architecture du projet

```
echosign/
├── config.py              # ⚙️  Configuration centrale (gestes, hyperparamètres)
├── keypoint_extractor.py  # 🖐️  Extraction MediaPipe Holistic (1662 keypoints/frame)
├── data_collector.py      # 📹  Enregistrement des gestes via webcam
├── train_model.py         # 🧠  Entraînement LSTM Bidirectionnel
├── recognizer.py          # ⚡  Reconnaissance en temps réel
├── main.py                # 🚀  Point d'entrée CLI
├── requirements.txt
├── dataset/               # Séquences numpy générées
└── models/                # Modèles .h5 + labels.npy
```

---

## ⚙️ Installation

```bash
# 1. Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

# 2. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation rapide

### Étape 1 — Configurer tes gestes
Ouvre `config.py` et modifie la liste `ACTIONS` :
```python
ACTIONS = [
    "bonjour",
    "merci",
    "oui",
    "non",
    "aide",
    # Ajoute tes gestes ici...
]
```

### Étape 2 — Collecter les données
```bash
python main.py --mode collect
```
- Un **compte à rebours** te laisse le temps de te préparer
- **30 séquences × 30 frames** par geste enregistrées
- Reprendre une collecte interrompue :
  ```bash
  python main.py --mode collect --resume
  ```

### Étape 3 — Entraîner le modèle
```bash
python main.py --mode train
```
- Courbes d'apprentissage et matrice de confusion générées automatiquement
- Le meilleur modèle est sauvegardé automatiquement

### Étape 4 — Reconnaître en temps réel
```bash
python main.py --mode recognize
```

### Voir l'état du projet
```bash
python main.py --mode status
```

---

## 🧠 Architecture du modèle

```
Input (30 frames × 1662 keypoints)
         ↓
BiLSTM(128) → Dropout(0.3) → BatchNorm
         ↓
BiLSTM(64)  → Dropout(0.3) → BatchNorm
         ↓
BiLSTM(64)  → Dropout(0.3) → BatchNorm
         ↓
Dense(128, relu) → Dropout(0.4)
         ↓
Dense(64, relu)
         ↓
Dense(n_classes, softmax)
```

---

## 📊 Keypoints MediaPipe Holistic

| Composant  | Points | Valeurs | Total  |
|------------|--------|---------|--------|
| Pose       | 33     | ×4      | 132    |
| Visage     | 468    | ×3      | 1404   |
| Main G.    | 21     | ×3      | 63     |
| Main D.    | 21     | ×3      | 63     |
| **Total**  |        |         | **1662** |

---

## 📱 Intégration Mobile (Flutter)
Le modèle peut être exporté en TensorFlow Lite pour l'app Flutter :
```python
import tensorflow as tf
model = tf.keras.models.load_model("models/echosign_lstm.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("models/echosign.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 🎯 Conseils pour un bon dataset
- **Éclairage** : luminosité uniforme, pas de contre-jour
- **Fond** : fond uni de préférence
- **Distance** : 50–80 cm de la webcam
- **Angle** : de face, mains bien visibles
- **Variété** : enregistre depuis plusieurs sessions pour la robustesse
