from __future__ import annotations
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

# ---- Paths inside the Airflow container ----
DATA_DIR = Path("/opt/airflow/data")
MODELS_DIR = Path("/opt/airflow/models")
INPUT_CSV = DATA_DIR / "messages.csv"                # expected columns: id,text[,label]
PREDICTIONS_CSV = DATA_DIR / "predictions_latest.csv"
METRICS_CSV = DATA_DIR / "metrics_history.csv"
MODEL_PATH = MODELS_DIR / "logreg_spam_pipeline.pkl"
BEST_THRESHOLD = 0.620

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 6, 1),
}

def ingest_data(**context):
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    context["ti"].xcom_push(key="n_rows", value=len(df))
    # keep it simple: persist raw for transparency (optional)
    (DATA_DIR / "ingested.csv").write_text(df.to_csv(index=False))

def score_model(**context):
    # Load model pipeline
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    with MODEL_PATH.open("rb") as f:
        pipeline = pickle.load(f)

    df = pd.read_csv(INPUT_CSV)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    probs = pipeline.predict_proba(df["text"].astype(str))[:, 1]
    preds = (probs >= BEST_THRESHOLD).astype(int)

    out = df.copy()
    out["prob_spam"] = probs
    out["pred"] = preds

    out.to_csv(PREDICTIONS_CSV, index=False)

def evaluate_metrics(**context):
    # only run if labels available
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"Predictions CSV not found at {PREDICTIONS_CSV}")

    df = pd.read_csv(PREDICTIONS_CSV)
    if "label" not in df.columns:
        # no labels -> just log + soft success
        print("No 'label' column found; skipping evaluation.")
        return

    y_true = df["label"].astype(int).values
    y_pred = df["pred"].astype(int).values
    y_prob = df["prob_spam"].astype(float).values

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    run_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    metrics_row = {
        "run_ts": run_ts,
        "n": len(df),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "auc": round(float(auc), 6) if auc == auc else None,  # nan-safe
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": BEST_THRESHOLD,
    }

    # Append to CSV (create if missing)
    if METRICS_CSV.exists():
        hist = pd.read_csv(METRICS_CSV)
        hist = pd.concat([hist, pd.DataFrame([metrics_row])], ignore_index=True)
    else:
        hist = pd.DataFrame([metrics_row])

    hist.to_csv(METRICS_CSV, index=False)
    print(f"Metrics appended to {METRICS_CSV}: {metrics_row}")

with DAG(
    dag_id="spam_batch_scoring",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,  # set "0 2 * * *" for daily 02:00
    catchup=False,
    description="Batch scoring + optional evaluation if labels exist",
    tags=["spam", "batch", "metrics"],
) as dag:

    t_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    t_score = PythonOperator(
        task_id="score_model",
        python_callable=score_model,
    )

    t_eval = PythonOperator(
        task_id="evaluate_metrics",
        python_callable=evaluate_metrics,
    )

    t_ingest >> t_score >> t_eval