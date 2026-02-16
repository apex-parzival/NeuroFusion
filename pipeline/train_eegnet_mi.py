"""
Train EEGNet (Keras) per-subject on BCI-IV-2a epochs.

Saves models to: models/eegnet/<SUBJ>_eegnet.h5
"""
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "processed" / "bci_iv_2a"
MODELS_DIR = PROJECT_ROOT / "models" / "eegnet"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- EEGNet builder (small, typical EEGNet-ish) ----------
def build_eegnet(n_channels, n_times, n_classes, F1=8, D=2, kern=64, dropoutRate=0.5):
    """
    Simple EEGNet-style architecture.
    Input shape: (n_channels, n_times, 1) or (n_channels, n_times)
    We'll expect channels-first: (n_channels, n_times, 1) via reshape in training code.
    """
    input_shape = (n_channels, n_times, 1)
    inp = layers.Input(shape=input_shape)

    # Temporal conv
    x = layers.Conv2D(F1, (1, kern), padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)

    # Depthwise conv (spatial)
    x = layers.DepthwiseConv2D((n_channels, 1), depth_multiplier=D, use_bias=False, depthwise_constraint=tf.keras.constraints.max_norm(1.))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropoutRate)(x)

    # Separable conv
    x = layers.SeparableConv2D(F1 * D, (1, 16), use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropoutRate)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------- Training loop ----------
def train_for_subject(subj, epochs=200, batch_size=32):
    Xp = PROCESSED / f"{subj}_X.npy"
    yp = PROCESSED / f"{subj}_y.npy"
    if not Xp.exists() or not yp.exists():
        print(f"[EEGNET] Missing for {subj}, skipping.")
        return None

    X = np.load(Xp)  # (n_trials, n_ch, n_times)
    y = np.load(yp).astype(int)

    n_trials, n_ch, n_t = X.shape
    print(f"[EEGNET] {subj} X shape {X.shape}")

    # Preprocess: reshape to Keras expected (batch, channels, times, 1)
    X = X.reshape((n_trials, n_ch, n_t, 1)).astype(np.float32)

    # One-hot labels
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    if Y.shape[1] == 1:  # binary -> transform to two columns
        Y = np.hstack([1 - Y, Y])

    # train/test split (subject-wise)
    Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    model = build_eegnet(n_ch, n_t, Y.shape[1])

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5)
    ]

    history = model.fit(Xtr, ytr, validation_data=(Xte, yte),
                        batch_size=batch_size, epochs=epochs, callbacks=cb, verbose=2)

    # save model and label binarizer
    model_path = MODELS_DIR / f"{subj}_eegnet.h5"
    meta_path = MODELS_DIR / f"{subj}_eegnet_meta.npz"
    model.save(model_path)
    np.savez(meta_path, classes=lb.classes_)
    print(f"[EEGNET] Saved {model_path}")

    # return test accuracy
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"[EEGNET] {subj} test acc: {acc*100:.2f}%")
    return acc

def main():
    subj_files = sorted(PROCESSED.glob("A??T_X.npy"))
    if not subj_files:
        raise FileNotFoundError("No subject epoch files found.")
    accs = []
    for p in subj_files:
        subj = p.stem.replace("_X", "")
        acc = train_for_subject(subj)
        if acc is not None:
            accs.append(acc)
    import numpy as np
    accs = np.array(accs)
    print("=== EEGNET SUMMARY ===")
    print(f"Mean: {accs.mean()*100:.2f}  Std: {accs.std()*100:.2f}")

if __name__ == "__main__":
    main()
