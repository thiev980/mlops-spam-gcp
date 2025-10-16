# dags/spam_eval.py
from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import os, json, base64
from io import BytesIO

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score,
    precision_score, recall_score, f1_score, roc_curve
)

# -------------------------------------------------------------------
# Pfade
# -------------------------------------------------------------------
DATA_DIR   = Path("/opt/airflow/data")
INCOMING   = DATA_DIR / "incoming"
LABELS     = DATA_DIR / "labels"
PREDS      = DATA_DIR / "predictions"
METRICS    = DATA_DIR / "metrics_history.csv"
THRESH_PATH = DATA_DIR / "threshold.json"

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 9, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

# -------------------------------------------------------------------
# Low-level Utils
# -------------------------------------------------------------------
def _atomic_write_df(df: pd.DataFrame, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)

def _atomic_write_bytes(content: bytes, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
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
            if s.min() >= -1e-9 and s.max() <= 1.0 + 1e-9:
                return c
    return None

def _load_threshold(default: float = 0.6) -> float:
    if THRESH_PATH.exists():
        try:
            return float(json.loads(THRESH_PATH.read_text()).get("threshold", default))
        except Exception:
            pass
    return default

def save_threshold(threshold: float, meta: dict | None = None):
    THRESH_PATH.write_text(json.dumps({"threshold": float(threshold), **(meta or {})}, indent=2))

# -------------------------------------------------------------------
# Threshold-Logik
# -------------------------------------------------------------------
def metrics_at_threshold(y_true, y_prob, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    # FPR (False-Positive-Rate)
    mask_ham = (y_true == 0)
    fp = np.sum((y_pred == 1) & mask_ham)
    tn = np.sum((y_pred == 0) & mask_ham)
    fpr = fp / (fp + tn + 1e-12)
    return {
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
    }

def pick_threshold(y_true, y_prob,
                   mode="precision",
                   target_precision=0.95,
                   target_recall=0.90,
                   target_fpr=0.001) -> tuple[float, dict]:
    """
    Returns: (best_threshold, summary_dict with precision/recall/f1/fpr at that threshold)
    Modi:
      - "f1"        : maximiere F1
      - "precision" : niedrigste Schwelle mit precision >= target_precision
      - "recall"    : größte Schwelle mit recall    >= target_recall
      - "fpr"       : größte Schwelle mit FPR       <= target_fpr
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if mode == "f1":
        prec, rec, thr = precision_recall_curve(y_true, y_prob)  # len(thr)=len(prec)-1
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        idx = np.argmax(f1[:-1]) if len(thr) else 0
        best_thr = float(thr[idx]) if len(thr) else 0.5
        return best_thr, metrics_at_threshold(y_true, y_prob, best_thr)

    elif mode == "precision":
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        candidates = [(t, p, r) for t, p, r in zip(thr, prec[:-1], rec[:-1]) if p >= target_precision]
        if candidates:
            # niedrigste Schwelle, die das Ziel hält -> mehr Recall
            t, _, _ = min(candidates, key=lambda x: x[0])
            return float(t), metrics_at_threshold(y_true, y_prob, float(t))
        # Fallback: Schwelle mit maximaler Precision
        if len(thr):
            idx = np.argmax(prec[:-1])
            t = float(thr[idx])
            return t, metrics_at_threshold(y_true, y_prob, t)
        return 0.5, metrics_at_threshold(y_true, y_prob, 0.5)

    elif mode == "recall":
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        candidates = [(t, p, r) for t, p, r in zip(thr, prec[:-1], rec[:-1]) if r >= target_recall]
        if candidates:
            # größte (strengste) Schwelle, die das Recall-Ziel noch erreicht
            t, _, _ = max(candidates, key=lambda x: x[0])
            return float(t), metrics_at_threshold(y_true, y_prob, float(t))
        # Fallback: F1
        return pick_threshold(y_true, y_prob, mode="f1")

    elif mode == "fpr":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        candidates = [(t, f) for t, f in zip(thr, fpr) if f <= target_fpr]
        if candidates:
            # größte (strengste) Schwelle unterhalb FPR-Grenze
            t, _ = max(candidates, key=lambda x: x[0])
            return float(t), metrics_at_threshold(y_true, y_prob, float(t))
        # Fallback: strengste Schwelle
        if len(thr):
            t = float(np.max(thr))
            return t, metrics_at_threshold(y_true, y_prob, t)
        return 0.5, metrics_at_threshold(y_true, y_prob, 0.5)

    else:
        raise ValueError("mode must be in {'f1','precision','recall','fpr'}")

# -------------------------------------------------------------------
# Kern: Auswertung + History schreiben
# -------------------------------------------------------------------
def _compute_metrics(ds: str):
    incoming_path = INCOMING / f"{ds}.csv"
    labels_path   = LABELS   / f"{ds}.csv"
    preds_path    = PREDS    / f"{ds}.csv"

    for p in [incoming_path, labels_path, preds_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    df_in  = pd.read_csv(incoming_path)   # id, text
    df_lab = pd.read_csv(labels_path)     # id, label
    df_pr  = pd.read_csv(preds_path)      # id, prediction, (proba_*)

    # Merge
    df = df_in.merge(df_lab, on="id", how="inner", suffixes=("", "_gt"))
    df = df.merge(df_pr, on="id", how="inner", suffixes=("", "_pred"))

    # Normalisieren
    if "label" in df.columns and "label_gt" not in df.columns:
        df = df.rename(columns={"label": "label_gt"})
    if "prediction" not in df.columns:
        for alt in ["pred", "hat", "label_pred"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "prediction"})
                break
    if "prediction" not in df.columns:
        raise ValueError("No prediction column found in predictions file")

    y_true = df["label_gt"].astype(int).to_numpy()
    y_pred = df["prediction"].astype(int).to_numpy()

    # Confusion + Metriken (basierend auf *verwendeter* Schwelle)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    eps = 1e-12
    accuracy     = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision    = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    recall       = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    specificity  = tn / max(tn + fp, 1) if (tn + fp) > 0 else 0.0
    f1           = 2 * precision * recall / max(precision + recall, eps) if (precision + recall) > 0 else 0.0
    balanced_acc = 0.5 * (recall + specificity)
    fpr_used     = fp / max(fp + tn, 1) if (fp + tn) > 0 else 0.0

    # ROC-AUC (falls Probas vorhanden)
    proba_col = _find_proba_col(df)
    roc_auc = None
    if proba_col is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, df[proba_col].astype(float).to_numpy()))
        except Exception:
            roc_auc = None

    # (a) Welche Schwelle wurde heute verwendet?
    thr_used = _load_threshold(0.6)

    # (b) Falls Probas vorhanden: neue *empfohlene* Schwelle berechnen (für morgen)
    suggested_thr = None
    suggested = {}
    if proba_col is not None:
        mode = os.getenv("THRESH_MODE", "precision").strip().lower()
        target_precision = float(os.getenv("THRESH_TARGET_PRECISION", "0.95"))
        target_recall    = float(os.getenv("THRESH_TARGET_RECALL", "0.90"))
        target_fpr       = float(os.getenv("THRESH_TARGET_FPR", "0.001"))

        best_thr, summary = pick_threshold(
            y_true, df[proba_col].astype(float).to_numpy(),
            mode=mode,
            target_precision=target_precision,
            target_recall=target_recall,
            target_fpr=target_fpr,
        )
        suggested_thr = float(best_thr)
        suggested = {"mode": mode,
                     "target_precision": target_precision,
                     "target_recall": target_recall,
                     "target_fpr": target_fpr,
                     **summary}
        # Speichern als Vorschlag für nächsten Tag (lokal)
        save_threshold(suggested_thr, meta={"suggested_from": ds, **suggested})

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
        "preds_file": str((PREDS / f"{ds}.csv").name),
        "threshold_used": float(thr_used),
        "fpr_used": round(float(fpr_used), 6),
        # vorgeschlagener Threshold (falls berechnet)
        "suggested_threshold": (float(suggested_thr) if suggested_thr is not None else None),
        "suggested_mode": suggested.get("mode"),
        "suggested_precision": suggested.get("precision"),
        "suggested_recall": suggested.get("recall"),
        "suggested_f1": suggested.get("f1"),
        "suggested_fpr": suggested.get("fpr"),
    }

    # Append/Upsert in CSV
    if METRICS.exists() and METRICS.stat().st_size > 0:
        hist = pd.read_csv(METRICS)
        if "ds" not in hist.columns and "date" in hist.columns:
            hist = hist.rename(columns={"date": "ds"})
        if "ds" not in hist.columns:
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        else:
            hist = hist[hist["ds"] != ds]
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        _atomic_write_df(hist, METRICS)
    else:
        _atomic_write_df(pd.DataFrame([row]), METRICS)

# -------------------------------------------------------------------
# Plot (Disk + kleines Thumbnail als Data URL für XCom/Inspect)
# -------------------------------------------------------------------
def plot_trend_png():
    import matplotlib.pyplot as plt

    if not METRICS.exists() or METRICS.stat().st_size == 0:
        print("No metrics yet; skipping plot.")
        return {"png_path": None, "data_url": None}

    df = pd.read_csv(METRICS)
    if "ds" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "ds"})
    if "ds" not in df.columns:
        print("No 'ds' column; skipping plot.")
        return {"png_path": None, "data_url": None}

    try:
        df["_ds_dt"] = pd.to_datetime(df["ds"])
        df = df.sort_values("_ds_dt")
        x = df["_ds_dt"]
    except Exception:
        df = df.sort_values("ds")
        x = df["ds"]

    fig, ax = plt.subplots(figsize=(8, 4))
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

    buf_full = BytesIO()
    fig.tight_layout()
    fig.savefig(buf_full, format="png", dpi=150)
    plt.close(fig)
    buf_full.seek(0)
    out = DATA_DIR / "metrics_trend.png"
    _atomic_write_bytes(buf_full.read(), out)
    print(f"[plot] wrote {out}")

    # kleines Thumbnail als Data URL (praktisch für XCom/Debug)
    fig_t, ax_t = plt.subplots(figsize=(5, 2.5))
    for col, label in [
        ("f1", "F1"),
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("balanced_accuracy", "Balanced Acc."),
        ("roc_auc", "ROC-AUC"),
    ]:
        if col in df.columns and df[col].notna().any():
            ax_t.plot(x, df[col], marker="o", label=label)
    ax_t.set_ylim(0, 1)
    ax_t.set_title("Trend")
    ax_t.grid(True, linestyle="--", alpha=0.4)
    ax_t.legend(fontsize="x-small", loc="lower right")
    buf_thumb = BytesIO()
    fig_t.tight_layout()
    fig_t.savefig(buf_thumb, format="png", dpi=110)
    plt.close(fig_t)
    buf_thumb.seek(0)
    b64 = base64.b64encode(buf_thumb.read()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    return {"png_path": str(out), "data_url": data_url}

# -------------------------------------------------------------------
# Vorschlag Threshold in Registry promoten
# -------------------------------------------------------------------
def promote_threshold_to_registry():
    import mlflow, json, os
    from pathlib import Path
    from mlflow.tracking import MlflowClient

    THRESH = Path("/opt/airflow/data/threshold.json")
    if not THRESH.exists():
        print("[promote] no local threshold.json -> nothing to promote")
        return

    # Datei lesen (validieren)
    try:
        data = json.loads(THRESH.read_text())
        thr = float(data.get("threshold"))
    except Exception as e:
        print(f"[promote] invalid threshold.json: {e}")
        return

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5002"))
    client = MlflowClient()
    MODEL_NAME = "spam-classifier"

    # aktuellen Production-Run holen
    vers = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not vers:
        print("[promote] no Production model version found -> skip")
        return
    run_id = vers[0].run_id

    # sichere Kopie in ein temp file und uploade nach model/threshold.json
    tmp = Path("/tmp/threshold.json")
    tmp.write_text(json.dumps({"threshold": thr}, indent=2))
    client.log_artifact(run_id=run_id, local_path=str(tmp), artifact_path="model")
    print(f"[promote] uploaded threshold={thr:.4f} to Production run {run_id} (model/threshold.json)")

# -------------------------------------------------------------------
# Upsert nach Postgres (Schema-Autoupdate)
# -------------------------------------------------------------------
def upsert_metrics_to_postgres():
    import psycopg2
    from psycopg2.extras import execute_values

    if not METRICS.exists() or METRICS.stat().st_size == 0:
        print("No metrics to upsert.")
        return

    df = pd.read_csv(METRICS)
    if "ds" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "ds"})
    if "ds" not in df.columns:
        raise ValueError("metrics_history.csv has no 'ds' column; cannot upsert")

    # DSN: eigenständige Metrik-DB oder Airflow-DB wiederverwenden (Demo)
    dsn = os.getenv("METRICS_PG_DSN") or os.getenv(
        "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
        "postgresql+psycopg2://airflow:airflow@postgres/airflow",
    )
    dsn = dsn.replace("+psycopg2", "")

    # Normierungen
    float_cols = [
        "accuracy","precision","recall","specificity","balanced_accuracy","f1",
        "roc_auc","spam_rate","fpr_used",
        "suggested_precision","suggested_recall","suggested_f1","suggested_fpr"
    ]
    int_cols = ["n","tp","tn","fp","fn"]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Build rows
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get("ds")),
            int((r.get("n") or 0)),
            int((r.get("tp") or 0)),
            int((r.get("tn") or 0)),
            int((r.get("fp") or 0)),
            int((r.get("fn") or 0)),
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
            float(r.get("threshold_used") or 0.0),
            float(r.get("fpr_used") or 0.0),
            (None if pd.isna(r.get("suggested_threshold")) else float(r.get("suggested_threshold"))),
            (None if pd.isna(r.get("suggested_mode")) else str(r.get("suggested_mode"))),
            (None if pd.isna(r.get("suggested_precision")) else float(r.get("suggested_precision"))),
            (None if pd.isna(r.get("suggested_recall")) else float(r.get("suggested_recall"))),
            (None if pd.isna(r.get("suggested_f1")) else float(r.get("suggested_f1"))),
            (None if pd.isna(r.get("suggested_fpr")) else float(r.get("suggested_fpr"))),
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
        preds_file TEXT,
        threshold_used DOUBLE PRECISION,
        fpr_used DOUBLE PRECISION,
        suggested_threshold DOUBLE PRECISION,
        suggested_mode TEXT,
        suggested_precision DOUBLE PRECISION,
        suggested_recall DOUBLE PRECISION,
        suggested_f1 DOUBLE PRECISION,
        suggested_fpr DOUBLE PRECISION
    );
    """
    alters = [
        "ALTER TABLE metrics_history ADD COLUMN IF NOT EXISTS threshold_used DOUBLE PRECISION;",
        "ALTER TABLE metrics_history ADD COLUMN IF NOT EXISTS fpr_used DOUBLE PRECISION;",
        "ALTER TABLE metrics_history ADD COLUMN IF NOT EXISTS suggested_threshold DOUBLE PRECISION;",
        "ALTER TABLE metrics_history ADD COLUMN IF NOT EXISTS suggested_mode TEXT;",
        "ALTER TABLE metrics_history ADD COLUMN IF NOT EXISTS suggested_precision DOUBLE PRECISION;",
        "ALTER TABLE metrics_history ADD COLUMN IF NOT EXISTS suggested_recall DOUBLE PRECISION;",
        "ALTER TABLE metrics_history ADD COLUMN IF NOT EXISTS suggested_f1 DOUBLE PRECISION;",
        "ALTER TABLE metrics_history ADD COLUMN IF NOT EXISTS suggested_fpr DOUBLE PRECISION;",
    ]

    insert_sql = """
    INSERT INTO metrics_history (
        ds, n, tp, tn, fp, fn, accuracy, precision, recall, specificity,
        balanced_accuracy, f1, roc_auc, spam_rate, proba_col, preds_file,
        threshold_used, fpr_used, suggested_threshold, suggested_mode,
        suggested_precision, suggested_recall, suggested_f1, suggested_fpr
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
        preds_file = EXCLUDED.preds_file,
        threshold_used = EXCLUDED.threshold_used,
        fpr_used = EXCLUDED.fpr_used,
        suggested_threshold = EXCLUDED.suggested_threshold,
        suggested_mode = EXCLUDED.suggested_mode,
        suggested_precision = EXCLUDED.suggested_precision,
        suggested_recall = EXCLUDED.suggested_recall,
        suggested_f1 = EXCLUDED.suggested_f1,
        suggested_fpr = EXCLUDED.suggested_fpr;
    """

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(create_sql)
        for stmt in alters:
            cur.execute(stmt)
        if rows:
            execute_values(cur, insert_sql, rows, page_size=500)
            print(f"[pg] upserted {len(rows)} rows into metrics_history")
        else:
            print("[pg] nothing to upsert")

# -------------------------------------------------------------------
# Task-Wrappers mit Orchestrator-ds
# -------------------------------------------------------------------
def eval_task(ds: str | None = None, **context):
    # Orchestrator-kompatibel: ds aus DagRun.conf übernehmen
    dag_run = context.get("dag_run")
    conf = (dag_run.conf if dag_run else {}) or {}
    if ds is None:
        ds = conf.get("ds")
    ds = ds or datetime.utcnow().strftime("%Y-%m-%d")
    return _compute_metrics(ds)

# -------------------------------------------------------------------
# DAG
# -------------------------------------------------------------------
with DAG(
    dag_id="spam_eval",
    default_args=DEFAULT_ARGS,
    schedule=None,            # vom Orchestrator getriggert
    catchup=False,
    tags=["spam", "eval"],
    description="Merge labels & predictions, compute metrics, append to history, plot trend, upsert Postgres (+suggest threshold).",
) as dag:
    evaluate  = PythonOperator(task_id="evaluate_and_append", python_callable=eval_task)
    plot_trend = PythonOperator(task_id="plot_trend_png", python_callable=plot_trend_png)
    upsert_pg  = PythonOperator(task_id="upsert_metrics_to_postgres", python_callable=upsert_metrics_to_postgres)

    promote = PythonOperator(task_id="promote_threshold_to_registry",
                         python_callable=promote_threshold_to_registry)

    evaluate >> plot_trend >> upsert_pg >> promote