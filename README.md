# Privacy-Preserving Two-Stage Intrusion Detection System using Federated Learning

A network intrusion detection system (IDS) trained with **federated learning**, so several organisations can build one shared model **without sharing their raw network traffic**. Only model weights leave each organisation.

Detection works in two stages:

1. **Stage 1:** is this network flow **benign** or an **attack**?
2. **Stage 2:** if it is an attack, **which type** is it (11 classes)?

The project includes the training notebooks, the trained models, and a small web UI with a FastAPI backend for trying the models on real network flows.

---

## Highlights

- Trained on **CICIDS2017** (2,830,743 flows, 78 flow features)
- **Federated training** with [Flower](https://flower.ai/) and **FedAvg** across 4 simulated clients
- **Non-IID clients:** same size, but attack share ranges from 5% to 43.8%, each with a different dominant attack type
- **Fair comparison:** the same DNN, optimiser and batch size for federated and centralized training
- **Demo UI:** edit the 78 features, load random real flows, and see the prediction from both stages

---

## Results (test set)

### Stage 1: Benign vs Attack (283,075 test flows)

| Metric    | Centralized | Federated |
|-----------|-------------|-----------|
| Accuracy  | 97.74%      | **97.90%** |
| Precision | **98.95%**  | 96.10%    |
| Recall    | 89.49%      | **93.11%** |
| F1-score  | 93.98%      | **94.58%** |

### Stage 2: Attack type (11 classes, 55,758 test flows)

| Metric (weighted) | Centralized | Federated |
|-------------------|-------------|-----------|
| Accuracy          | **99.77%**  | 99.43%    |
| Precision         | **99.68%**  | 99.36%    |
| F1-score          | **99.72%**  | 99.37%    |

**Key finding:** federated training matches the centralized model in Stage 1 and comes within 0.4 points in Stage 2, without sharing any raw traffic. Rare classes are the weak point: under federated training, Bot recall drops to 66%, and Web Attack XSS is not detected by either model.

---

## Architecture

```
          Client 1        Client 2        Client 3        Client 4
          5% attack       10% attack      20% attack      43.8% attack
              │               │               │               │
              └──── local training, send weights only ────────┘
                                      │
                            Flower server (FedAvg)
                                      │
                                Global DNN model
                                      │
   Incoming flow ──► Stage 1: Benign / Attack ──► Stage 2: Attack type ──► Alert
```

**Model (both stages):** fully connected DNN with 3 hidden layers

| Stage | Layers | Output | Parameters |
|-------|--------|--------|------------|
| Stage 1 | 78 → 128 → 64 → 32 → 1 | Sigmoid (attack probability) | 20,481 |
| Stage 2 | 78 → 128 → 64 → 32 → 11 | Softmax (attack type) | 20,811 |

**Training setup**

| Setting | Federated | Centralized |
|---------|-----------|-------------|
| Data | 4 non-IID clients | Pooled training set |
| Training | 3 rounds × 1 local epoch | 3 epochs |
| Optimiser | Adam, lr = 0.001 | Adam, lr = 0.001 |
| Batch size | 1,024 | 1,024 |
| Aggregation | FedAvg (weighted by sample count) | – |

**Attack classes (Stage 2):** Bot, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, PortScan, SSH-Patator, Web Attack – Brute Force, Web Attack – XSS

---

## Project structure

```
IDSPRESERVING/
├── app/                      # Demo web UI + backend
│   ├── server.py             # FastAPI backend (two-stage inference)
│   ├── build_samples.py      # Builds the sample bank from the test sets
│   ├── sample_bank.npz       # 4,011 real flows used by "Refresh values"
│   ├── requirements.txt
│   └── static/               # Frontend (index.html, style.css, app.js)
├── data/processed/
│   ├── feature_names.txt     # The 78 feature names
│   ├── scaler.joblib         # Stage 1 StandardScaler
│   └── stage2/
│       ├── label_mapping.txt # Attack class index → name
│       └── scaled/scaler.joblib  # Stage 2 StandardScaler
├── models/
│   ├── stage1/               # fl_final_model.pt, centralized_final_model.pt
│   └── stage2/               # federated_final_model.pt, centralized_final_model.pt
├── notebooks/                # Training notebooks (Google Colab)
├── results/                  # Metrics (JSON/CSV) and confusion matrices
├── src/
│   ├── model.py              # IDSModel (PyTorch)
│   └── fl/                   # Flower client and server
└── run_ui.bat                # Start the demo on Windows
```

> **Note:** the processed dataset arrays (`.npy`, `.csv`) are not included because they are too large for GitHub (the training set alone is about 700 MB). The demo UI does not need them.

---

## Running the demo UI

**Requirements:** Python 3.10 or newer

```bash
# 1. Install dependencies (first time only)
pip install -r app/requirements.txt

# 2. Start the backend + UI from the project root
python -m uvicorn app.server:app --port 8000
```

On Windows you can also double-click `run_ui.bat`.

Then open **http://127.0.0.1:8000** in your browser.

### Using the UI

- **72 fields** are pre-filled with a real network flow from the test sets. You can edit any of them (edited fields turn yellow).
- **6 fields** are left empty for you to enter: Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets, Flow Bytes/s, Flow Packets/s. The grey hint shows the real value, and **Fill empty fields** copies the real values in.
- **Refresh values** loads another random flow. You can choose any flow, benign only, attack only, or a specific attack type.
- **Predict** shows **BENIGN** or **ATTACK**, plus the attack type and class probabilities.
- **Model** switches between the federated and centralized models.

To change which features are left empty, edit `EMPTY_FEATURES` at the top of `app/server.py`.

### API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/meta` | Feature names, attack labels, available models, test metrics |
| GET  | `/api/sample?kind=any\|benign\|attack\|<attack type>` | A random real flow and its true label |
| POST | `/api/predict` | Run the two-stage prediction on 78 feature values |
| GET  | `/api/health` | Health check |

Interactive API docs: **http://127.0.0.1:8000/docs**

Example request:

```json
POST /api/predict
{
  "model": "federated",
  "features": { "Destination Port": 80, "Flow Duration": 4, "...": "all 78 features" }
}
```

---

## Training

The models were trained in Google Colab (Tesla T4 GPU) using the notebooks in `notebooks/`:

1. Load CICIDS2017 (8 CSV files), clean infinite/missing values and encode labels
2. Stratified 80/10/10 train/validation/test split
3. Fit a `StandardScaler` on the training split only
4. Split the training data into 4 non-IID clients
5. Train with Flower (FedAvg) and train a centralized baseline with the same settings
6. Repeat on attack-only data (11 classes) for Stage 2

---

## Known limitations

- Stage 1 rarely flags **Bot** and **Web Attack** flows because they are rare in the training data, so in the demo these often come back as benign.
- Results are from a single run of 3 federated rounds.
- Weight sharing alone is not fully private (gradients can leak information). Secure aggregation or differential privacy is planned as future work.

## Future work

- More federated rounds and **FedProx** to reduce client drift
- Class weighting for rare classes (Bot, web attacks)
- Secure aggregation / differential privacy
- Migration to the newer `flwr run` workflow

---

## Tech stack

PyTorch · Flower · scikit-learn · NumPy · FastAPI · Uvicorn · HTML/CSS/JavaScript

## Dataset

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP 2018. [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

## Author

**Subhajit Mahapatra** (25MCA0062)
Department of Computer Applications, School of Computer Science Engineering and Information Systems, VIT
Guide: **Dr. Vijay Anand R**, Assistant Professor Sr. Grade 1, SCORE
