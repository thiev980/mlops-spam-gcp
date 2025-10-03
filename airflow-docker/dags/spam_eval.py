# airflow_docker/dags/spam_eval.py
from __future__ import annotations
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import os
from io import BytesIO

DATA_DIR = Path("/opt/airflow/data")
INCOMING = DATA_DIR / "incoming"
LABELS   = DATA_DIR / "labels"
PREDS    = DATA_DIR / "predictions"
METRICS  = DATA_DIR / "metrics_history.csv"

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 9, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

def _atomic_write_df(df: pd.DataFrame, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)

def _atomic_write_bytes(content: bytes, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(content)
    tmp.replace(dest)

def _find_proba_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "proba_spam", "probability_spam", "spam_proba", "proba", "p_spam", "score_spam"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Heuristik: irgendeine float-Spalte zwischen 0..1
    for c in df.columns:
        s = df[c]
        if np.issubdtype(s.dtype, np.floating):
            if s.min() >= 0.0 - 1e-9 and s.max() <= 1.0 + 1e-9:
                return c
    return None

def _compute_metrics(ds: str):
    incoming_path = INCOMING / f"{ds}.csv"
    labels_path   = LABELS   / f"{ds}.csv"
    preds_path    = PREDS    / f"{ds}.csv"

    # Sanity
    for p in [incoming_path, labels_path, preds_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    df_in  = pd.read_csv(incoming_path)   # id, text
    df_lab = pd.read_csv(labels_path)     # id, label
    df_pr  = pd.read_csv(preds_path)      # id, prediction, (proba_*)

    # Merge
    df = df_in.merge(df_lab, on="id", how="inner", suffixes=("", "_gt"))
    df = df.merge(df_pr, on="id", how="inner", suffixes=("", "_pred"))

    # Spalten normalisieren
    if "label" in df.columns and "label_gt" not in df.columns:
        df = df.rename(columns={"label": "label_gt"})
    if "prediction" not in df.columns:
        # fallback: pred/hat/label_pred
        for alt in ["pred", "hat", "label_pred"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "prediction"})
                break
    if "prediction" not in df.columns:
        raise ValueError("No prediction column found in predictions file")

    y_true = df["label_gt"].astype(int).to_numpy()
    y_pred = df["prediction"].astype(int).to_numpy()

    # Confusion
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    eps = 1e-12
    accuracy    = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision   = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    recall      = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0  # TPR / Sensitivity
    specificity = tn / max(tn + fp, 1) if (tn + fp) > 0 else 0.0  # TNR
    f1          = 2 * precision * recall / max(precision + recall, eps) if (precision + recall) > 0 else 0.0
    balanced_acc= 0.5 * (recall + specificity)

    # ROC-AUC (falls Probas vorhanden)
    proba_col = _find_proba_col(df)
    roc_auc = None
    if proba_col is not None:
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = float(roc_auc_score(y_true, df[proba_col].astype(float).to_numpy()))
        except Exception:
            roc_auc = None

    # Input-Spamrate (Ground truth)
    spam_rate = float((y_true == 1).mean()) if len(y_true) else 0.0

    row = {
        "ds": ds,
        "n": int(len(df)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "balanced_accuracy": round(balanced_acc, 6),
        "f1": round(f1, 6),
        "roc_auc": (round(roc_auc, 6) if roc_auc is not None else None),
        "spam_rate": round(spam_rate, 6),
        "proba_col": proba_col,
        "preds_file": str(preds_path.name),
    }

    # Append/Upsert nach metrics_history.csv (legacy-sicher)
    if METRICS.exists() and METRICS.stat().st_size > 0:
        hist = pd.read_csv(METRICS)

        # Migration/Kompatibilität:
        # 1) Falls alte Datei 'date' statt 'ds' hat -> umbenennen
        if "ds" not in hist.columns and "date" in hist.columns:
            hist = hist.rename(columns={"date": "ds"})

        # 2) Falls weiterhin keine 'ds'-Spalte existiert, können wir kein Upsert machen -> einfach anhängen
        if "ds" not in hist.columns:
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        else:
            # Normales Upsert pro ds
            hist = hist[hist["ds"] != ds]
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)

        _atomic_write_df(hist, METRICS)
    else:
        _atomic_write_df(pd.DataFrame([row]), METRICS)

def plot_trend_png():
    import matplotlib.pyplot as plt

    if not METRICS.exists() or METRICS.stat().st_size == 0:
        print("No metrics yet; skipping plot.")
        return

    df = pd.read_csv(METRICS)
    # tolerate legacy column names
    if "ds" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "ds"})

    if "ds" not in df.columns:
        print("No 'ds' column; skipping plot.")
        return

    # sort by ds; try to parse dates for nicer x-axis
    try:
        df["_ds_dt"] = pd.to_datetime(df["ds"])
        df = df.sort_values("_ds_dt")
        x = df["_ds_dt"]
    except Exception:
        df = df.sort_values("ds")
        x = df["ds"]

    fig, ax = plt.subplots(figsize=(8, 4))
    # only plot if present
    for col, label in [
        ("f1", "F1"),
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("balanced_accuracy", "Balanced Acc."),
        ("roc_auc", "ROC-AUC"),
    ]:
        if col in df.columns and df[col].notna().any():
            ax.plot(x, df[col], marker="o", label=label)

    ax.set_ylim(0, 1)
    ax.set_title("Spam Model – Metric Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    out = DATA_DIR / "metrics_trend.png"
    _atomic_write_bytes(buf.read(), out)
    print(f"[plot] wrote {out}")

def upsert_metrics_to_postgres():
    import psycopg2
    from psycopg2.extras import execute_values

    if not METRICS.exists() or METRICS.stat().st_size == 0:
        print("No metrics to upsert.")
        return

    df = pd.read_csv(METRICS)
    # normalize ds column
    if "ds" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "ds"})
    if "ds" not in df.columns:
        raise ValueError("metrics_history.csv has no 'ds' column; cannot upsert")

    # Connection string:
    # prefer explicit METRICS_PG_DSN, else reuse Airflow's DB (works fine for demo)
    dsn = os.getenv("METRICS_PG_DSN") or os.getenv(
        "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
        "postgresql+psycopg2://airflow:airflow@postgres/airflow",
    )
    # psycopg2 expects no '+psycopg2' driver fragment
    dsn = dsn.replace("+psycopg2", "")

    # ensure numeric columns are proper types (avoid NaN casting issues)
    numeric_cols = [
        "n","tp","tn","fp","fn","accuracy","precision","recall","specificity",
        "balanced_accuracy","f1","roc_auc","spam_rate"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get("ds")),
            int(r.get("n") or 0),
            int(r.get("tp") or 0),
            int(r.get("tn") or 0),
            int(r.get("fp") or 0),
            int(r.get("fn") or 0),
            float(r.get("accuracy") or 0.0),
            float(r.get("precision") or 0.0),
            float(r.get("recall") or 0.0),
            float(r.get("specificity") or 0.0),
            float(r.get("balanced_accuracy") or 0.0),
            float(r.get("f1") or 0.0),
            (None if pd.isna(r.get("roc_auc")) else float(r.get("roc_auc"))),
            float(r.get("spam_rate") or 0.0),
            (None if pd.isna(r.get("proba_col")) else str(r.get("proba_col"))),
            (None if pd.isna(r.get("preds_file")) else str(r.get("preds_file"))),
        ))

    create_sql = """
    CREATE TABLE IF NOT EXISTS metrics_history (
        ds TEXT PRIMARY KEY,
        n INTEGER,
        tp INTEGER, tn INTEGER, fp INTEGER, fn INTEGER,
        accuracy DOUBLE PRECISION,
        precision DOUBLE PRECISION,
        recall DOUBLE PRECISION,
        specificity DOUBLE PRECISION,
        balanced_accuracy DOUBLE PRECISION,
        f1 DOUBLE PRECISION,
        roc_auc DOUBLE PRECISION,
        spam_rate DOUBLE PRECISION,
        proba_col TEXT,
        preds_file TEXT
    );
    """

    insert_sql = """
    INSERT INTO metrics_history (
        ds, n, tp, tn, fp, fn, accuracy, precision, recall, specificity,
        balanced_accuracy, f1, roc_auc, spam_rate, proba_col, preds_file
    ) VALUES %s
    ON CONFLICT (ds) DO UPDATE SET
        n = EXCLUDED.n,
        tp = EXCLUDED.tp,
        tn = EXCLUDED.tn,
        fp = EXCLUDED.fp,
        fn = EXCLUDED.fn,
        accuracy = EXCLUDED.accuracy,
        precision = EXCLUDED.precision,
        recall = EXCLUDED.recall,
        specificity = EXCLUDED.specificity,
        balanced_accuracy = EXCLUDED.balanced_accuracy,
        f1 = EXCLUDED.f1,
        roc_auc = EXCLUDED.roc_auc,
        spam_rate = EXCLUDED.spam_rate,
        proba_col = EXCLUDED.proba_col,
        preds_file = EXCLUDED.preds_file;
    """

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(create_sql)
        execute_values(cur, insert_sql, rows, page_size=500)
    print(f"[pg] upserted {len(rows)} rows into metrics_history")

def eval_task(ds: str, **_):
    return _compute_metrics(ds)

with DAG(
    dag_id="spam_eval",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,
    catchup=False,
    tags=["spam","eval"],
    description="Merge labels & predictions, compute metrics, append to history, plot trend, upsert Postgres",
) as dag:
    evaluate = PythonOperator(task_id="evaluate_and_append", python_callable=eval_task)
    plot_trend = PythonOperator(task_id="plot_trend_png", python_callable=plot_trend_png)
    upsert_pg = PythonOperator(task_id="upsert_metrics_to_postgres", python_callable=upsert_metrics_to_postgres)

    evaluate >> plot_trend >> upsert_pg