"""
Hybrid fusion training per-subject.

Tries to load:
  processed/deap/<SUBJ>_emo_X.npy  (per-trial emotion features)
If available and lengths match, trains EEGNet + MLP fusion.
Otherwise falls back to EEGNet-only and saves a hybrid wrapper (EEG-only).
Saves to models/hybrid/<SUBJ>_hybrid.h5 and meta npz.
"""
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_MI = PROJECT_ROOT / "processed" / "bci_iv_2a"
PROCESSED_EMO = PROJECT_ROOT / "processed" / "deap"
MODELS_DIR = PROJECT_ROOT / "models" / "hybrid"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# reuse EEGNet builder from earlier (simple version)
def build_eegnet_branch(n_channels, n_times, F1=8, D=2, kern=64, dropoutRate=0.5):
    inp = layers.Input(shape=(n_channels, n_times, 1))
    x = layers.Conv2D(F1, (1, kern), padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((n_channels, 1), depth_multiplier=D, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropoutRate)(x)
    x = layers.SeparableConv2D(F1 * D, (1, 16), use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropoutRate)(x)
    x = layers.Flatten()(x)
    return inp, x

def build_emo_branch(n_features):
    inp = layers.Input(shape=(n_features,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    return inp, x

def build_fusion_model(n_ch, n_t, n_emo_feats, n_classes):
    eeg_inp, eeg_out = build_eegnet_branch(n_ch, n_t)
    emo_inp, emo_out = build_emo_branch(n_emo_feats)
    merged = layers.concatenate([eeg_out, emo_out])
    x = layers.Dense(128, activation='elu')(merged)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=[eeg_inp, emo_inp], outputs=out)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_hybrid_for_subject(subj, epochs=200, batch_size=32):
    Xp = PROCESSED_MI / f"{subj}_X.npy"
    yp = PROCESSED_MI / f"{subj}_y.npy"
    if not Xp.exists():
        print(f"[HYBRID] No MI data for {subj}, skipping")
        return None

    Xmi = np.load(Xp)  # (n_trials, n_ch, n_t)
    y = np.load(yp).astype(int)
    n_trials, n_ch, n_t = Xmi.shape

    # try to load paired emotion features
    emo_path = PROCESSED_EMO / f"{subj}_emo_X.npy"
    use_emo = emo_path.exists()
    Xemo = None
    if use_emo:
        Xemo = np.load(emo_path)
        if Xemo.shape[0] != n_trials:
            print(f"[HYBRID] Found emo features but trial counts differ for {subj} ({Xemo.shape[0]} vs {n_trials}). Ignoring emo for now.")
            use_emo = False

    # prepare labels
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])

    # reshape MI for EEGNet
    Xmi_in = Xmi.reshape((n_trials, n_ch, n_t, 1)).astype(np.float32)

    # split
    idx = np.arange(n_trials)
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    Xmi_tr, Xmi_te = Xmi_in[tr_idx], Xmi_in[te_idx]
    ytr, yte = Y[tr_idx], Y[te_idx]

    if use_emo:
        Xemo_tr, Xemo_te = Xemo[tr_idx], Xemo[te_idx]
        model = build_fusion_model(n_ch, n_t, Xemo.shape[1], Y.shape[1])
        history = model.fit([Xmi_tr, Xemo_tr], ytr, validation_data=([Xmi_te, Xemo_te], yte),
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                            verbose=2)
    else:
        # no emo: build tiny fusion where emo branch is zeros
        # alternatively, we can just train EEGNet-only and save under hybrid path
        model = build_eegnet_branch(n_ch, n_t)[0]  # placeholder to avoid error
        # Instead: build a small EEGNet-only final model
        from tensorflow.keras import models as kmodels
        inp = layers.Input(shape=(n_ch, n_t, 1))
        # reuse small EEGNet body from EEGNet builder:
        eeg_inp, eeg_out = build_eegnet_branch(n_ch, n_t)
        out = layers.Dense(Y.shape[1], activation='softmax')(eeg_out)
        model = kmodels.Model(inputs=eeg_inp, outputs=out)
        model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(Xmi_tr, ytr, validation_data=(Xmi_te, yte),
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                            verbose=2)

    # save
    model_path = MODELS_DIR / f"{subj}_hybrid.h5"
    meta_path = MODELS_DIR / f"{subj}_hybrid_meta.npz"
    model.save(model_path)
    np.savez(meta_path, classes=lb.classes_, used_emo=use_emo)
    loss, acc = model.evaluate(Xmi_te if not use_emo else [Xmi_te, Xemo_te], yte, verbose=0)
    print(f"[HYBRID] {subj} test acc: {acc*100:.2f}% (used_emo={use_emo})")
    return acc

def main():
    subj_files = sorted(PROCESSED_MI.glob("A??T_X.npy"))
    if not subj_files:
        raise FileNotFoundError("No MI epoch files found.")
    accs = []
    for p in subj_files:
        subj = p.stem.replace("_X", "")
        acc = train_hybrid_for_subject(subj, epochs=150, batch_size=32)
        if acc is not None:
            accs.append(acc)
    import numpy as np
    accs = np.array(accs)
    print("=== HYBRID SUMMARY ===")
    print(f"Mean: {accs.mean()*100:.2f}  Std: {accs.std()*100:.2f}")

if __name__ == "__main__":
    main()
