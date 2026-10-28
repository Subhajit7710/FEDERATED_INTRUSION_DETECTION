# IDS Model Demo UI

A small web UI + FastAPI backend for the two-stage intrusion detection model.

- **Stage 1** (`models/stage1/*.pt`, `data/processed/scaler.joblib`): BENIGN vs ATTACK
- **Stage 2** (`models/stage2/*.pt`, `data/processed/stage2/scaled/scaler.joblib`): which of the 11 attack types
  (only run when stage 1 says ATTACK)

## Run

From the `IDSPRESERVING` folder:

```
pip install -r app/requirements.txt
python -m uvicorn app.server:app --port 8000
```

(or double-click `run_ui.bat` on Windows), then open http://127.0.0.1:8000

## Using it

- 72 features are pre-filled with a real flow from the test sets and can be edited (edited fields turn yellow).
- 6 features are left empty for you to type (the placeholder shows the sample's real value).
  **Fill empty fields** copies those real values in.
- **Refresh values** loads another random real flow. *Sample from* picks any / benign / attack / a specific attack type.
- **Predict** sends all 78 values to the backend and shows BENIGN or ATTACK, plus the attack type and probabilities.
- If you didn't change anything, the result is compared against the sample's true label.

To choose which 6 features are empty, edit `EMPTY_FEATURES` at the top of `app/server.py`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/meta` | feature names, empty features, attack labels, models, test metrics |
| GET  | `/api/sample?kind=any\|benign\|attack\|<attack label>` | one random real flow (raw values) + its true label |
| POST | `/api/predict` | body `{"features": {"<name>": value, ... all 78}, "model": "federated"\|"centralized"}` |
| GET  | `/api/health` | health check |

Interactive API docs: http://127.0.0.1:8000/docs

## Sample bank

`app/sample_bank.npz` holds 4,011 real flows (2,000 benign from the stage 1 test set, up to 200 per attack type
from the stage 2 test set). Rebuild it with `python app/build_samples.py`.

## Known model behaviour

Stage 1 rarely flags **Bot** and **Web Attack** flows (they are rare in training), so those samples usually come back
BENIGN. This matches the stage 1 test metrics (recall ~93%) and is not a UI issue.
