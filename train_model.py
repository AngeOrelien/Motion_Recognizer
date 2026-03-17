"""
EchoSign Vision - Entraînement du modèle LSTM
==============================================
Charge le dataset numpy, entraîne un réseau LSTM bidirectionnel
et sauvegarde le modèle + les labels.

Usage :
    python train_model.py
    python train_model.py --epochs 300 --batch 16
"""

import os
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Réduit les logs TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     BatchNormalization, Bidirectional)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, TensorBoard)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from config import (
    DATA_DIR, MODEL_PATH, LABEL_PATH, LOG_DIR,
    ACTIONS, NB_SEQUENCES, SEQUENCE_LENGTH, KEYPOINT_SIZE,
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, LEARNING_RATE
)


# ── Chargement du dataset ─────────────────────────────────────────────────────

def load_dataset(actions=None):
    """
    Charge toutes les séquences numpy depuis DATA_DIR.
    Retourne X (N, SEQUENCE_LENGTH, KEYPOINT_SIZE) et y (N,) labels str.
    """
    actions = actions or ACTIONS
    X, y = [], []
    missing = []

    print("\n  Chargement du dataset...")
    for action in actions:
        action_dir = Path(DATA_DIR) / action
        found = 0
        for seq_idx in range(NB_SEQUENCES):
            seq_path = action_dir / f"{seq_idx}.npy"
            if seq_path.exists():
                sequence = np.load(str(seq_path))
                if sequence.shape == (SEQUENCE_LENGTH, KEYPOINT_SIZE):
                    X.append(sequence)
                    y.append(action)
                    found += 1
                else:
                    print(f"  ⚠️  {seq_path.name} : shape inattendue {sequence.shape}")
            else:
                missing.append(str(seq_path))

        status = "✓" if found == NB_SEQUENCES else f"⚠ {found}/{NB_SEQUENCES}"
        print(f"    [{status:>12}]  {action}")

    if missing:
        print(f"\n  ⚠️  {len(missing)} fichiers manquants (dataset incomplet).")

    return np.array(X), np.array(y)


# ── Construction du modèle ────────────────────────────────────────────────────

def build_model(n_classes: int) -> tf.keras.Model:
    """
    Réseau LSTM Bidirectionnel pour la classification de séquences.
    Architecture :
      BiLSTM(128) → Dropout → BiLSTM(64) → Dropout → BiLSTM(64) →
      BatchNorm → Dense(128, relu) → Dropout → Dense(n_classes, softmax)
    """
    model = Sequential([
        # ── Bloc 1 : extraction des patterns temporels ──────────────────
        Bidirectional(LSTM(128, return_sequences=True, activation='tanh'),
                      input_shape=(SEQUENCE_LENGTH, KEYPOINT_SIZE)),
        Dropout(0.3),
        BatchNormalization(),

        # ── Bloc 2 ──────────────────────────────────────────────────────
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
        Dropout(0.3),
        BatchNormalization(),

        # ── Bloc 3 : contexte final ──────────────────────────────────────
        Bidirectional(LSTM(64, return_sequences=False, activation='tanh')),
        Dropout(0.3),
        BatchNormalization(),

        # ── Classifieur ─────────────────────────────────────────────────
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_history(history, save_path: str = None):
    """Affiche et sauvegarde les courbes d'apprentissage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EchoSign Vision — Courbes d'Entraînement", fontsize=14, weight='bold')

    # Précision
    axes[0].plot(history.history['accuracy'],     label='Train', color='royalblue')
    axes[0].plot(history.history['val_accuracy'], label='Val',   color='coral')
    axes[0].set_title('Précision')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Perte
    axes[1].plot(history.history['loss'],     label='Train', color='royalblue')
    axes[1].plot(history.history['val_loss'], label='Val',   color='coral')
    axes[1].set_title('Perte (Loss)')
    axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Courbes sauvegardées : {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels, save_path: str = None):
    """Affiche la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title('EchoSign — Matrice de Confusion')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Matrice sauvegardée : {save_path}")
    plt.show()


# ── Entraînement ──────────────────────────────────────────────────────────────

def train(epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Pipeline complet : chargement → modèle → entraînement → évaluation."""

    print("\n" + "=" * 60)
    print("   EchoSign Vision  —  Entraînement du Modèle")
    print("=" * 60)

    # 1. Chargement des données
    X, y_str = load_dataset()
    if len(X) == 0:
        print("\n❌ Aucune donnée trouvée. Lance d'abord data_collector.py !")
        return

    # 2. Encodage des labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    y_cat = to_categorical(y_enc)
    labels = le.classes_

    np.save(LABEL_PATH, labels)
    print(f"\n  Labels ({len(labels)}) : {list(labels)}")
    print(f"  Dataset : {X.shape}  —  {len(labels)} classes")

    # 3. Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y_enc
    )
    print(f"  Train : {len(X_train)}  |  Test : {len(X_test)}")

    # 4. Construction du modèle
    model = build_model(len(labels))
    model.summary()

    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=15, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    ]

    # 6. Entraînement
    print(f"\n  Démarrage de l'entraînement ({epochs} époques max)...\n")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Évaluation
    print("\n" + "=" * 60)
    print("  ÉVALUATION SUR LE JEU DE TEST")
    print("=" * 60)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss     : {loss:.4f}")
    print(f"  Accuracy : {acc * 100:.2f}%")

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = le.inverse_transform(np.argmax(y_pred_proba, axis=1))
    y_true = le.inverse_transform(np.argmax(y_test, axis=1))

    print("\n" + classification_report(y_true, y_pred, target_names=labels))

    # 8. Visualisations
    model_dir = Path(MODEL_PATH).parent
    plot_history(history, save_path=str(model_dir / "training_curves.png"))
    plot_confusion_matrix(y_true, y_pred, labels,
                          save_path=str(model_dir / "confusion_matrix.png"))

    print(f"\n  ✅ Modèle sauvegardé : {MODEL_PATH}")
    print(f"  ✅ Labels sauvegardés : {LABEL_PATH}")
    print("\n  Lance maintenant : python main.py --mode recognize")


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoSign - Entraînement")
    parser.add_argument("--epochs",  type=int, default=EPOCHS,     help="Nombre d'époques")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE, help="Taille du batch")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch)
