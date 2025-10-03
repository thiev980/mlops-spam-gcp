# spam_score_model.py
from __future__ import annotations
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd, pickle

DATA_DIR   = Path("/opt/airflow/data")
INCOMING   = DATA_DIR / "incoming"
PRED_DIR   = DATA_DIR / "predictions"
PRED_LATEST= DATA_DIR / "predictions_latest.csv"
MODEL_PATH = Path("/opt/airflow/models/logreg_spam_pipeline.pkl")
BEST_THRESHOLD = 0.620
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

def score_model(ds: str, **_):
    in_path = INCOMING / f"{ds}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    with MODEL_PATH.open("rb") as f:
        pipeline = pickle.load(f)

    df = pd.read_csv(in_path)
    probs = pipeline.predict_proba(df["text"].astype(str))[:, 1]
    preds = (probs >= BEST_THRESHOLD).astype(int)
    out = df.copy()
    out["prob_spam"] = probs
    out["pred"] = preds

    pred_path = PRED_DIR / f"{ds}.csv"
    _atomic_write(out, pred_path)
    _atomic_write(out, PRED_LATEST)
    print(f"Wrote predictions → {pred_path} & {PRED_LATEST}")

with DAG(
    dag_id="spam_score_model",
    default_args=DEFAULT_ARGS,
    schedule_interval="5 2 * * *",   # 02:05 täglich (nach Generierung)
    catchup=False,
    tags=["spam","scoring"],
    description="Score daily messages with the model",
) as dag:
    t = PythonOperator(task_id="score_model", python_callable=score_model)