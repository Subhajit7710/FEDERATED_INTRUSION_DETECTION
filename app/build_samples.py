"""
Build a small bank of real network flows used by the UI's "Refresh" button.

Benign rows come from the Stage 1 test set (stored scaled, so they are
inverse-transformed back to raw values). Attack rows come from the Stage 2
test set, which is stored unscaled together with the 11-class labels.

Run once from the project root:
    python app/build_samples.py
"""

from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent / "sample_bank.npz"

BENIGN_ROWS = 2000
ATTACK_ROWS_PER_CLASS = 200
SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    # ---- Benign (Stage 1 test set, scaled -> raw) ----
    X1 = np.load(DATA / "scaled" / "X_test.npy", mmap_mode="r")
    y1 = np.load(DATA / "scaled" / "y_test.npy")
    benign_idx = np.flatnonzero(y1 == 0)
    pick = np.sort(rng.choice(benign_idx, size=BENIGN_ROWS, replace=False))
    scaler1 = joblib.load(DATA / "scaler.joblib")
    benign = scaler1.inverse_transform(np.asarray(X1[pick], dtype=np.float64))
    benign = np.round(benign, 4)
    benign[np.abs(benign) < 1e-6] = 0.0

    # ---- Attacks (Stage 2 test set, already raw) ----
    X2 = np.load(DATA / "stage2" / "X_test.npy")
    y2 = np.load(DATA / "stage2" / "y_test.npy")
    attack_rows, attack_labels = [], []
    for cls in np.unique(y2):
        idx = np.flatnonzero(y2 == cls)
        n = min(ATTACK_ROWS_PER_CLASS, len(idx))
        sel = rng.choice(idx, size=n, replace=False)
        attack_rows.append(X2[sel].astype(np.float64))
        attack_labels.append(np.full(n, cls, dtype=np.int64))
    attacks = np.vstack(attack_rows)
    attack_labels = np.concatenate(attack_labels)

    # label -1 = benign, 0..10 = attack class index
    X = np.vstack([benign, attacks])
    y = np.concatenate([np.full(len(benign), -1, dtype=np.int64), attack_labels])

    np.savez_compressed(OUT, X=X, y=y)
    print(f"Saved {len(X)} flows to {OUT}")
    print(f"  benign: {len(benign)}")
    for cls, count in zip(*np.unique(attack_labels, return_counts=True)):
        print(f"  attack class {cls}: {count}")


if __name__ == "__main__":
    main()
