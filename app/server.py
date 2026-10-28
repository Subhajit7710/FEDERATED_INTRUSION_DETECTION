"""
IDS inference server.

Two-stage pipeline:
  Stage 1  binary DNN  (78 -> 1, sigmoid)   BENIGN vs ATTACK
  Stage 2  multiclass  (78 -> 11, softmax)  attack type, only run when Stage 1 says ATTACK

Each stage has its own StandardScaler, fitted on its own training data.

Start from the project root:
    python -m uvicorn app.server:app --port 8000
then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Dict, List, Literal, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Paths and configuration
# --------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Features left empty in the UI for the user to type in.
# Change this list to pick different ones (names must match feature_names.txt).
EMPTY_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
]

MODEL_FILES = {
    "federated": {
        "stage1": MODELS / "stage1" / "fl_final_model.pt",
        "stage2": MODELS / "stage2" / "federated_final_model.pt",
        "metrics1": RESULTS / "stage1" / "fl_test_metrics.json",
        "metrics2": RESULTS / "stage2" / "fl_test_metrics.json",
    },
    "centralized": {
        "stage1": MODELS / "stage1" / "centralized_final_model.pt",
        "stage2": MODELS / "stage2" / "centralized_final_model.pt",
        "metrics1": RESULTS / "stage1" / "centralized_test_metrics.json",
        "metrics2": RESULTS / "stage2" / "centralized_test_metrics.json",
    },
}

ATTACK_THRESHOLD = 0.5

torch.set_num_threads(1)


# --------------------------------------------------------------------------
# Model definition (same layout as src/model.py, output size configurable)
# --------------------------------------------------------------------------

class IDSModel(nn.Module):
    def __init__(self, input_size: int = 78, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.network(x)


def load_model(path: Path) -> IDSModel:
    state = torch.load(path, map_location="cpu", weights_only=True)
    out_size = state["network.6.weight"].shape[0]
    in_size = state["network.0.weight"].shape[1]
    model = IDSModel(in_size, out_size)
    model.load_state_dict(state)
    model.eval()
    return model


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def read_attack_labels() -> List[str]:
    text = (DATA / "stage2" / "label_mapping.txt").read_bytes().decode("utf-8", errors="replace")
    labels = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        idx, name = line.split(":", 1)
        # the source file has a broken en-dash in "Web Attack – Brute Force"
        name = name.strip().replace("�", "-").replace("\x96", "-")
        name = " ".join(name.split())
        labels[int(idx)] = name
    return [labels[i] for i in sorted(labels)]


# --------------------------------------------------------------------------
# Load everything once at startup
# --------------------------------------------------------------------------

FEATURES: List[str] = [
    f.strip() for f in (DATA / "feature_names.txt").read_text().splitlines() if f.strip()
]
N_FEATURES = len(FEATURES)
assert N_FEATURES == 78, f"expected 78 features, found {N_FEATURES}"
for name in EMPTY_FEATURES:
    assert name in FEATURES, f"EMPTY_FEATURES contains unknown feature: {name}"

ATTACK_LABELS = read_attack_labels()

SCALER_STAGE1 = joblib.load(DATA / "scaler.joblib")
SCALER_STAGE2 = joblib.load(DATA / "stage2" / "scaled" / "scaler.joblib")

PIPELINES = {}
for key, files in MODEL_FILES.items():
    if files["stage1"].exists() and files["stage2"].exists():
        PIPELINES[key] = {
            "stage1": load_model(files["stage1"]),
            "stage2": load_model(files["stage2"]),
            "metrics1": read_json(files["metrics1"]),
            "metrics2": read_json(files["metrics2"]),
        }
if not PIPELINES:
    raise RuntimeError("No model files found under models/stage1 and models/stage2")
DEFAULT_MODEL = "federated" if "federated" in PIPELINES else next(iter(PIPELINES))

_bank_path = APP_DIR / "sample_bank.npz"
if not _bank_path.exists():
    from app.build_samples import main as _build  # noqa: E402
    _build()
_bank = np.load(_bank_path)
BANK_X: np.ndarray = _bank["X"]
BANK_Y: np.ndarray = _bank["y"]  # -1 benign, 0..10 attack class

_rng = np.random.default_rng()
_rng_lock = threading.Lock()


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def predict_flow(values: np.ndarray, model_key: str) -> dict:
    pipe = PIPELINES[model_key]
    x = values.reshape(1, -1).astype(np.float64)

    x1 = np.nan_to_num(SCALER_STAGE1.transform(x), nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        logit = pipe["stage1"](torch.tensor(x1, dtype=torch.float32)).item()
    p_attack = 1.0 / (1.0 + math.exp(-logit)) if logit > -700 else 0.0

    result = {
        "model": model_key,
        "attack_probability": p_attack,
        "threshold": ATTACK_THRESHOLD,
        "verdict": "ATTACK" if p_attack >= ATTACK_THRESHOLD else "BENIGN",
        "attack_type": None,
        "attack_type_confidence": None,
        "attack_type_probabilities": None,
    }

    if result["verdict"] == "ATTACK":
        x2 = np.nan_to_num(SCALER_STAGE2.transform(x), nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            probs = torch.softmax(pipe["stage2"](torch.tensor(x2, dtype=torch.float32)), dim=1)[0]
        probs = probs.numpy().astype(float)
        order = np.argsort(probs)[::-1]
        result["attack_type"] = ATTACK_LABELS[int(order[0])]
        result["attack_type_confidence"] = float(probs[order[0]])
        result["attack_type_probabilities"] = [
            {"label": ATTACK_LABELS[int(i)], "probability": float(probs[i])} for i in order
        ]

    return result


def label_name(y: int) -> str:
    return "BENIGN" if y < 0 else ATTACK_LABELS[int(y)]


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

app = FastAPI(title="IDS Model Demo", version="1.0")


class PredictRequest(BaseModel):
    features: Dict[str, Optional[float]]
    model: Optional[str] = None


@app.get("/api/meta")
def meta():
    return {
        "features": FEATURES,
        "empty_features": EMPTY_FEATURES,
        "attack_labels": ATTACK_LABELS,
        "models": list(PIPELINES.keys()),
        "default_model": DEFAULT_MODEL,
        "threshold": ATTACK_THRESHOLD,
        "metrics": {
            k: {"stage1": v["metrics1"], "stage2": v["metrics2"]} for k, v in PIPELINES.items()
        },
        "sample_bank": {
            "benign": int((BANK_Y < 0).sum()),
            "attack": int((BANK_Y >= 0).sum()),
        },
    }


@app.get("/api/sample")
def sample(kind: str = "any"):
    """Return one real flow from the test sets.

    kind: "any" | "benign" | "attack" | an attack label (e.g. "DDoS")
    """
    if kind == "any":
        pool = np.arange(len(BANK_Y))
    elif kind == "benign":
        pool = np.flatnonzero(BANK_Y < 0)
    elif kind == "attack":
        pool = np.flatnonzero(BANK_Y >= 0)
    elif kind in ATTACK_LABELS:
        pool = np.flatnonzero(BANK_Y == ATTACK_LABELS.index(kind))
    else:
        raise HTTPException(400, f"unknown sample kind: {kind}")
    if len(pool) == 0:
        raise HTTPException(404, "no samples of that kind")

    with _rng_lock:
        i = int(_rng.choice(pool))
    row = BANK_X[i]
    return {
        "index": i,
        "true_label": label_name(int(BANK_Y[i])),
        "features": {name: float(v) for name, v in zip(FEATURES, row)},
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    model_key = req.model or DEFAULT_MODEL
    if model_key not in PIPELINES:
        raise HTTPException(400, f"unknown model: {model_key}")

    unknown = [k for k in req.features if k not in FEATURES]
    if unknown:
        raise HTTPException(422, {"message": "Unknown feature names", "features": unknown})

    missing, invalid = [], []
    values = np.zeros(N_FEATURES, dtype=np.float64)
    for i, name in enumerate(FEATURES):
        v = req.features.get(name)
        if v is None:
            missing.append(name)
        elif not math.isfinite(v):
            invalid.append(name)
        else:
            values[i] = v
    if missing or invalid:
        raise HTTPException(
            422,
            {
                "message": "All 78 features need a finite numeric value",
                "missing": missing,
                "invalid": invalid,
            },
        )

    return predict_flow(values, model_key)


@app.get("/api/health")
def health():
    return {"status": "ok", "models": list(PIPELINES.keys())}


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")
