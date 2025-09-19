from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import pickle

MODEL_PATH = "/opt/airflow/models/logreg_spam_pipeline.pkl"
INPUT_PATH = "/opt/airflow/data/messages.csv"       # Eingabe-CSV
OUTPUT_DIR = "/opt/airflow/data"                    # wohin Ergebnisse gehen
BEST_THRESHOLD = 0.620

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 6, 1),
}

def ingest_data(**context):
    # hier könntest du auch aus S3/DB lesen; wir lesen lokal
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    # einfache Validierung
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    # Zwischenablage (optional)
    tmp_path = os.path.join(OUTPUT_DIR, "ingested.parquet")
    df.to_parquet(tmp_path, index=False)
    context["ti"].xcom_push(key="ingested_path", value=tmp_path)

def preprocess_data(**context):
    # in deinem Fall: Rohtext direkt an Pipeline (damit Preprocessing matcht)
    # Wir kopieren nur durch; wenn du extra Preprocessing willst, tu's hier.
    ingested_path = context["ti"].xcom_pull(key="ingested_path")
    df = pd.read_parquet(ingested_path)
    preproc_path = os.path.join(OUTPUT_DIR, "preprocessed.parquet")
    df.to_parquet(preproc_path, index=False)
    context["ti"].xcom_push(key="preprocessed_path", value=preproc_path)

def score_model(execution_date=None, **context):
    preproc_path = context["ti"].xcom_pull(key="preprocessed_path")
    df = pd.read_parquet(preproc_path)

    # Modell laden
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    # Score berechnen
    probs = pipeline.predict_proba(df["text"].astype(str).tolist())
    spam_prob = probs[:, 1]
    preds = (spam_prob >= BEST_THRESHOLD).astype(int)
    labels = ["ham" if p == 0 else "spam" for p in preds]

    out = df.copy()
    out["probability_spam"] = spam_prob
    out["prediction"] = labels

    # Ausgabedatei (mit Datum)
    ds = (execution_date or datetime.utcnow()).strftime("%Y-%m-%d")
    out_path = os.path.join(OUTPUT_DIR, f"predictions_{ds}.csv")
    out.to_csv(out_path, index=False)

with DAG(
    dag_id="spam_batch_scoring",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,   # manuell starten; stell z.B. "0 2 * * *" für täglich um 02:00 ein
    catchup=False,
    tags=["mlops", "batch", "spam"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
        provide_context=True,
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        provide_context=True,
    )

    score = PythonOperator(
        task_id="score_model",
        python_callable=score_model,
        provide_context=True,
    )

    ingest >> preprocess >> score
