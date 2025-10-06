# spam_train_tune.py
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import os, json
import pandas as pd
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_recall_curve, roc_auc_score, precision_score, recall_score
)
import mlflow
import mlflow.sklearn
import pickle
import optuna
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb


DATA_DIR   = Path("/opt/airflow/data")
INCOMING   = DATA_DIR / "incoming"
LABELS     = DATA_DIR / "labels"
MODELS_DIR = Path("/opt/airflow/models")
CURRENT_MODEL = MODELS_DIR / "model_current.pkl"
THRESH_PATH   = DATA_DIR / "threshold.json"

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 9, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

def _atomic_write_bytes(b: bytes, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(b)
    tmp.replace(dest)

def _atomic_write_text(txt: str, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_text(txt)
    tmp.replace(dest)

def collect_training_data(min_days: int = 3) -> pd.DataFrame:
    # sammelt alle ds-Dateien, joint incoming + labels pro Tag und konkateniert
    if not INCOMING.exists() or not LABELS.exists():
        raise FileNotFoundError("incoming/ oder labels/ fehlen")

    pairs = []
    for lab in sorted(LABELS.glob("*.csv")):
        ds = lab.stem
        inc = INCOMING / f"{ds}.csv"
        if not inc.exists():
            continue
        df_lab = pd.read_csv(lab)
        df_in  = pd.read_csv(inc)
        df = df_in.merge(df_lab, on="id", how="inner")  # id, text, label
        if {"text","label"}.issubset(df.columns):
            pairs.append(df[["text","label"]])

    if not pairs:
        raise RuntimeError("Keine Trainingsdaten gefunden (incoming + labels leer?)")

    df_all = pd.concat(pairs, ignore_index=True).dropna(subset=["text","label"])
    if df_all["label"].nunique() < 2:
        raise RuntimeError("Trainingsdaten enthalten nur eine Klasse – mehr Daten sammeln.")
    if len(pairs) < min_days:
        print(f"[warn] nur {len(pairs)} Tag(e) Trainingsdaten – läuft dennoch.")
    return df_all

def pick_threshold(y_true, y_prob, mode="precision", target_precision=0.90):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)  # len(thr)=len(prec)-1
    # Wir optimieren F1 unter der Nebenbedingung 'precision >= target_precision'
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    candidates = [(t,p,r,f1i) for t,p,r,f1i in zip(thr, prec[:-1], rec[:-1], f1[:-1]) if p >= target_precision]
    if candidates:
        # nimm den mit bestem F1 unter den Precision-Constraint
        t,p,r,f1i = max(candidates, key=lambda x: x[3])
        return float(t), float(p), float(r), float(f1i)
    # Fallback: bestes F1 insgesamt
    idx = int(np.argmax(f1[:-1]))
    return float(thr[idx]), float(prec[idx]), float(rec[idx]), float(f1[idx])


# ---------------------------
# Optuna-Objective mit 3 Modellen
# ---------------------------
def objective_factory(X_train, y_train):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    import optuna

    def objective(trial: optuna.Trial):
        model_type = trial.suggest_categorical("model_type", ["logreg", "linear_svc", "xgb"])
        ngram_hi   = trial.suggest_int("ngram_hi", 1, 2)
        max_feat   = trial.suggest_int("max_features", 4000, 30000, step=4000)

        if model_type == "logreg":
            C = trial.suggest_float("logreg_C", 1e-2, 1e+2, log=True)
            clf = LogisticRegression(C=C, max_iter=300, solver="liblinear")
        elif model_type == "linear_svc":
            C = trial.suggest_float("svm_C", 1e-3, 1e+2, log=True)
            # Für CV-F1 reicht ungecalibrierte LinearSVC:
            clf = LinearSVC(C=C)
        else:  # xgb
            n_estimators = trial.suggest_int("xgb_n_estimators", 200, 700, step=100)
            max_depth    = trial.suggest_int("xgb_max_depth", 3, 8)
            lr           = trial.suggest_float("xgb_eta", 0.02, 0.3, log=True)
            subsample    = trial.suggest_float("xgb_subsample", 0.6, 1.0)
            colsample    = trial.suggest_float("xgb_colsample", 0.6, 1.0)
            # xgboost kann sparse CSR direkt
            clf = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                subsample=subsample,
                colsample_bytree=colsample,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=1,
            )

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_hi), max_features=max_feat, min_df=2)),
            ("clf", clf),
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        return float(scores.mean())
    return objective

def tune_and_train():
    # MLflow wie gehabt ...
    mlruns_path = "/opt/airflow/mlruns"
    os.makedirs(mlruns_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment("spam_tuning")

    df = collect_training_data()
    X = df["text"].astype(str).to_numpy()
    y = df["label"].astype(int).to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    import optuna
    study = optuna.create_study(direction="maximize")
    with mlflow.start_run(run_name="optuna_search"):
        study.optimize(objective_factory(X_train, y_train), n_trials=40, show_progress_bar=False)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("cv_f1_best", study.best_value)

    # -------- bestes Modell rekonstruieren & final fit --------
    p = study.best_params
    model_type = p["model_type"]
    ngram_hi   = p["ngram_hi"]
    max_feat   = p["max_features"]

    if model_type == "logreg":
        clf = LogisticRegression(C=p["logreg_C"], max_iter=400, solver="liblinear")
        needs_calibration = False
    elif model_type == "linear_svc":
        # Für Threshold brauchst du Probas -> kalibrieren!
        base = LinearSVC(C=p["svm_C"])
        clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
        needs_calibration = True
    else:
        clf = xgb.XGBClassifier(
            n_estimators=p["xgb_n_estimators"],
            max_depth=p["xgb_max_depth"],
            learning_rate=p["xgb_eta"],
            subsample=p["xgb_subsample"],
            colsample_bytree=p["xgb_colsample"],
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=1,
        )
        needs_calibration = False

    pipe_best = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_hi), max_features=max_feat, min_df=2)),
        ("clf", clf),
    ])

    with mlflow.start_run(run_name="final_fit"):
        pipe_best.fit(X_train, y_train)

        # Val-Probas (LinearSVC via Calibration liefert predict_proba)
        if hasattr(pipe_best.named_steps["clf"], "predict_proba"):
            val_prob = pipe_best.predict_proba(X_val)[:, 1]
        else:
            # Fallback: decision_function -> Sigmoid
            from sklearn.preprocessing import MinMaxScaler
            dec = pipe_best.decision_function(X_val)
            # simple scaling to 0..1 (zur Not)
            val_prob = MinMaxScaler().fit_transform(dec.reshape(-1,1)).ravel()

        thr, p_at_thr, r_at_thr, f1_at_thr = pick_threshold(y_val, val_prob, mode="precision", target_precision=0.90)
        y_hat = (val_prob >= thr).astype(int)

        f1  = f1_score(y_val, y_hat)
        prec= precision_score(y_val, y_hat, zero_division=0)
        rec = recall_score(y_val, y_hat, zero_division=0)
        try:
            auc = roc_auc_score(y_val, val_prob)
        except Exception:
            auc = None

        mlflow.log_params({"model_type": model_type, "ngram_hi": ngram_hi, "max_features": max_feat})
        mlflow.log_metric("val_f1", f1)
        mlflow.log_metric("val_precision", prec)
        mlflow.log_metric("val_recall", rec)
        if auc is not None:
            mlflow.log_metric("val_roc_auc", auc)
        mlflow.log_metric("chosen_threshold", thr)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CURRENT_MODEL, "wb") as f:
            pickle.dump(pipe_best, f)
        mlflow.log_artifact(str(CURRENT_MODEL), artifact_path="model")

        _atomic_write_text(json.dumps({"threshold": float(thr)}, indent=2), THRESH_PATH)
        mlflow.log_artifact(str(THRESH_PATH), artifact_path="model")

    print(f"[train] model_type={model_type}")
    print(f"[train] wrote model → {CURRENT_MODEL}")
    print(f"[train] wrote threshold → {THRESH_PATH}")

with DAG(
    dag_id="spam_train_tune",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,     # manuell oder via Orchestration (vor Score/Eval)
    catchup=False,
    tags=["spam","train","mlflow","optuna"],
    description="Collect daily labeled data, Optuna tune, log to MLflow, save best model + threshold.",
) as dag:

    train = PythonOperator(
        task_id="tune_and_train",
        python_callable=tune_and_train
    )