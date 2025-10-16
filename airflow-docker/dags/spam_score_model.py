# dags/spam_score_model.py
from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import os, json

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


# --------------------------------------------------------------------
# Pfade & Konstanten
# --------------------------------------------------------------------
DATA_DIR    = Path("/opt/airflow/data")
INCOMING    = DATA_DIR / "incoming"
PRED_DIR    = DATA_DIR / "predictions"
PRED_LATEST = DATA_DIR / "predictions_latest.csv"
THRESH_PATH = DATA_DIR / "threshold.json"

PRED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "spam-classifier"   # Registry-Name

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 9, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _atomic_write(df: pd.DataFrame, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)


def _load_threshold_from_registry() -> float:
    """
    Reihenfolge:
      1) ENV THRESH_FORCE (z. B. für Tests)
      2) threshold.json aus dem Production-Run (MLflow Registry)
      3) Lokales threshold.json (Fallback)
      4) ENV THRESH_DEFAULT oder 0.6
    """
    # 1) Harte Forcierung
    tf = os.getenv("THRESH_FORCE")
    if tf:
        try:
            t = float(tf)
            print(f"[threshold] using THRESH_FORCE={t}")
            return t
        except Exception:
            print(f"[threshold] invalid THRESH_FORCE={tf}, ignoring.")

    # 2) Aus Registry (Production)
    try:
        client = MlflowClient()
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not prod_versions:
            raise RuntimeError("No Production model found in registry.")

        run_id = prod_versions[0].run_id
        # Wir erwarten threshold.json auf Artefakt-Pfad "model/threshold.json".
        # Download robust via Client (liefert lokalen Ordnerpfad zurück).
        local_dir = client.download_artifacts(run_id, "model")
        thresh_file = Path(local_dir) / "threshold.json"
        if thresh_file.exists():
            obj = json.loads(thresh_file.read_text())
            t = float(obj.get("threshold"))
            print(f"[threshold] using registry threshold.json={t}")
            return t
        else:
            print("[threshold] threshold.json not found in Production artifacts.")
    except Exception as e:
        print(f"[threshold] failed to read threshold from registry: {e}")

    # 3) Lokales threshold.json
    if THRESH_PATH.exists():
        try:
            obj = json.loads(THRESH_PATH.read_text())
            t = float(obj.get("threshold"))
            print(f"[threshold] using local threshold.json={t}")
            return t
        except Exception as e:
            print(f"[threshold] failed to read local threshold.json: {e}")

    # 4) Default
    tdef = os.getenv("THRESH_DEFAULT", "0.6")
    try:
        t = float(tdef)
    except Exception:
        t = 0.6
    print(f"[threshold] using default={t}")
    return t


def score_model(ds: str | None = None, **context):
    # DagRun conf (falls via TriggerDagRunOperator gesetzt)
    dag_run = context.get("dag_run")
    conf = (dag_run.conf if dag_run else {}) or {}

    if ds is None:
        ds = conf.get("ds")
    ds = ds or datetime.utcnow().strftime("%Y-%m-%d")

    in_path = INCOMING / f"{ds}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    # MLflow Tracking-URI (aus docker-compose gesetzt)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5002"))

    # Modell aus Registry laden (Stage: Production)
    model_uri = f"models:/{MODEL_NAME}/Production"
    try:
        pipeline = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_uri}: {e}")

    df = pd.read_csv(in_path)

    # Textspalte robust finden
    text_col = next((c for c in ["text", "message", "body", "content"] if c in df.columns), None)
    if text_col is None:
        raise ValueError(
            f"No text column found in {in_path}. "
            "Expected one of ['text','message','body','content']."
        )

    if len(df) == 0:
        out = df.copy()
        out["proba_spam"] = []
        out["prediction"] = []
        _atomic_write(out, PRED_DIR / f"{ds}.csv")
        _atomic_write(out, PRED_LATEST)
        print(f"[scoring] empty input -> wrote empty predictions for {ds}")
        return

    X = df[text_col].astype(str)

    # Wahrscheinlichkeiten + Threshold
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)[:, 1]
    else:
        # Fallback: Scores auf 0..1 minmaxen (selten benötigt)
        from sklearn.preprocessing import MinMaxScaler
        scores = pipeline.decision_function(X)
        probs = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()

    thr = _load_threshold_from_registry()
    preds = (probs >= thr).astype(int)

    out = df.copy()
    out["proba_spam"] = probs
    out["prediction"] = preds

    pred_path = PRED_DIR / f"{ds}.csv"
    _atomic_write(out, pred_path)
    _atomic_write(out, PRED_LATEST)
    print(f"[scoring] model={model_uri} threshold_used={thr:.4f} → wrote {pred_path.name} & {PRED_LATEST.name}")


# --------------------------------------------------------------------
# Airflow DAG
# --------------------------------------------------------------------
with DAG(
    dag_id="spam_score_model",
    default_args=DEFAULT_ARGS,
    schedule=None,                 # wird vom Orchestrator getriggert
    catchup=False,
    tags=["spam", "scoring"],
    description="Score daily messages with the model (loads Production from MLflow Registry).",
) as dag:
    t = PythonOperator(
        task_id="score_model",
        python_callable=score_model,
    )