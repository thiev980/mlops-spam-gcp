# airflow_docker/dags/spam_score_model.py
from __future__ import annotations
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import os, json, pickle
import pandas as pd

DATA_DIR    = Path("/opt/airflow/data")
INCOMING    = DATA_DIR / "incoming"
PRED_DIR    = DATA_DIR / "predictions"
PRED_LATEST = DATA_DIR / "predictions_latest.csv"
MODEL_PATH  = Path("/opt/airflow/models/logreg_spam_pipeline.pkl")
THRESH_PATH = DATA_DIR / "threshold.json"

PRED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 9, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

def _atomic_write(df: pd.DataFrame, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)

def _load_threshold() -> float:
    """
    Reihenfolge:
      1) ENV THRESH_FORCE (falls gesetzt) -> immer verwenden
      2) threshold.json (falls vorhanden & gültig)
      3) ENV THRESH_DEFAULT (sonst 0.6)
    """
    # 1) harte Forcierung (z. B. für Tests)
    tf = os.getenv("THRESH_FORCE")
    if tf:
        try:
            t = float(tf)
            print(f"[threshold] using THRESH_FORCE={t}")
            return t
        except Exception:
            print(f"[threshold] invalid THRESH_FORCE={tf}, ignoring.")

    # 2) aus Datei (vom Eval vorgeschlagen)
    if THRESH_PATH.exists():
        try:
            obj = json.loads(THRESH_PATH.read_text())
            t = float(obj.get("threshold"))
            print(f"[threshold] using threshold.json={t}")
            return t
        except Exception as e:
            print(f"[threshold] failed to read threshold.json: {e}")

    # 3) Default aus ENV oder 0.6
    tdef = os.getenv("THRESH_DEFAULT", "0.6")
    try:
        t = float(tdef)
    except Exception:
        t = 0.6
    print(f"[threshold] using default={t}")
    return t

def score_model(ds: str, **_):
    in_path = INCOMING / f"{ds}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    # Modell laden
    with MODEL_PATH.open("rb") as f:
        pipeline = pickle.load(f)

    df = pd.read_csv(in_path)

    # Textspalte robust finden
    text_col = None
    for c in ["text", "message", "body", "content"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"No text column found in {in_path}. Expected one of ['text','message','body','content'].")

    # leere Inputs behandeln
    if len(df) == 0:
        out = df.copy()
        out["proba_spam"] = []
        out["prediction"] = []
        _atomic_write(out, PRED_DIR / f"{ds}.csv")
        _atomic_write(out, PRED_LATEST)
        print(f"[scoring] empty input -> wrote empty predictions for {ds}")
        return

    probs = pipeline.predict_proba(df[text_col].astype(str))[:, 1]
    thr = _load_threshold()
    preds = (probs >= thr).astype(int)

    out = df.copy()
    out["proba_spam"] = probs
    out["prediction"] = preds

    pred_path = PRED_DIR / f"{ds}.csv"
    _atomic_write(out, pred_path)
    _atomic_write(out, PRED_LATEST)
    print(f"[scoring] threshold_used={thr:.4f} → wrote {pred_path.name} & {PRED_LATEST.name}")

with DAG(
    dag_id="spam_score_model",
    default_args=DEFAULT_ARGS,
    schedule_interval="5 2 * * *",   # 02:05 täglich (nach Generierung)
    catchup=False,
    tags=["spam","scoring"],
    description="Score daily messages with the model (reads threshold.json, writes proba_spam/prediction).",
) as dag:
    t = PythonOperator(task_id="score_model", python_callable=score_model)