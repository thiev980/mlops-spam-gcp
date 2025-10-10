# smoke_score.py
import os, json
import mlflow
import mlflow.sklearn
import pandas as pd

MODEL_NAME = "spam-classifier"

# MLflow Tracking-URI (so wie in docker-compose)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002"))

print(f"[smoke] loading model '{MODEL_NAME}' from registry (alias=production)...")
model_uri = f"models:/{MODEL_NAME}@production"
pipeline = mlflow.sklearn.load_model(model_uri)

# Beispiel-Messages
messages = [
    "Claim your FREE prize now! Visit winner[.]com",
    "Hey, just confirming our meeting tomorrow at 10am",
    "Update your bank login immediately to avoid suspension",
    "Family dinner tonight 🍝 shall we meet at 7?"
]
df = pd.DataFrame({"text": messages})

# Probabilities
probs = pipeline.predict_proba(df["text"])[:, 1]

# Threshold laden (aus Registry-Artefakt oder Fallback)
from pathlib import Path
from mlflow.tracking import MlflowClient

client = mlflow.tracking.MlflowClient()
run_id = client.get_model_version_by_alias(MODEL_NAME, "production").run_id
local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
thresh_file = Path(local_dir) / "threshold.json"
if thresh_file.exists():
    thr = float(json.loads(thresh_file.read_text())["threshold"])
else:
    thr = 0.6

preds = (probs >= thr).astype(int)

print("\n=== Smoke-Test Results ===")
for msg, p, pred in zip(messages, probs, preds):
    print(f"{p:.3f} → {pred} | {msg}")
print(f"[threshold used] {thr}")