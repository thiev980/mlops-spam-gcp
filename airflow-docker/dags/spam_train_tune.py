# spam_train_tune.py
from __future__ import annotations

import os, json, pickle, traceback, warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ---- laute/irrelevante Warnungen wegdämpfen (optional) -----------------------
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")  # unterdrückt Git-Python Warnung
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------------------------------------------------------
# Pfade & Konstanten
# --------------------------------------------------------------------
DATA_DIR   = Path("/opt/airflow/data")
INCOMING   = DATA_DIR / "incoming"
LABELS     = DATA_DIR / "labels"
MODELS_DIR = Path("/opt/airflow/models")
CURRENT_MODEL = MODELS_DIR / "model_current.pkl"
THRESH_PATH   = DATA_DIR / "threshold.json"

MODEL_NAME = "spam-classifier"   # Name im MLflow Model Registry

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 9, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _atomic_write_text(txt: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_text(txt)
    tmp.replace(dest)

def collect_training_data(min_days: int = 3) -> pd.DataFrame:
    """Sammelt alle Tagesdateien (incoming+labels), joint sie und konkateniert."""
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
        if {"text", "label"}.issubset(df.columns):
            pairs.append(df[["text", "label"]])

    if not pairs:
        raise RuntimeError("Keine Trainingsdaten gefunden (incoming + labels leer?)")

    df_all = pd.concat(pairs, ignore_index=True).dropna(subset=["text", "label"])
    if df_all["label"].nunique() < 2:
        raise RuntimeError("Trainingsdaten enthalten nur eine Klasse – mehr Daten sammeln.")
    if len(pairs) < min_days:
        print(f"[warn] nur {len(pairs)} Tag(e) Trainingsdaten – läuft dennoch.")
    return df_all

def pick_threshold(y_true, y_prob, target_precision=0.90):
    """Wählt eine Schwelle: bestes F1 unter Nebenbedingung precision >= target_precision."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)  # len(thr)=len(prec)-1
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    candidates = [
        (t, p, r, f1i)
        for t, p, r, f1i in zip(thr, prec[:-1], rec[:-1], f1[:-1])
        if p >= target_precision
    ]
    if candidates:
        t, p, r, f1i = max(candidates, key=lambda x: x[3])
        return float(t), float(p), float(r), float(f1i)
    idx = int(np.argmax(f1[:-1]))
    return float(thr[idx]), float(prec[idx]), float(rec[idx]), float(f1[idx])

# --------------------------------------------------------------------
# Optuna-Objective mit 3 Modellen (LogReg, LinearSVC, XGBoost)
# --------------------------------------------------------------------
def objective_factory(X_train, y_train):
    def objective(trial: optuna.Trial):
        model_type = trial.suggest_categorical("model_type", ["logreg", "linear_svc", "xgb"])
        ngram_hi   = trial.suggest_int("ngram_hi", 1, 2)
        max_feat   = trial.suggest_int("max_features", 4000, 30000, step=4000)

        if model_type == "logreg":
            C = trial.suggest_float("logreg_C", 1e-2, 1e+2, log=True)
            clf = LogisticRegression(C=C, max_iter=300, solver="liblinear")
        elif model_type == "linear_svc":
            C = trial.suggest_float("svm_C", 1e-3, 1e+2, log=True)
            clf = LinearSVC(C=C)
        else:  # xgb
            n_estimators = trial.suggest_int("xgb_n_estimators", 200, 700, step=100)
            max_depth    = trial.suggest_int("xgb_max_depth", 3, 8)
            lr           = trial.suggest_float("xgb_eta", 0.02, 0.3, log=True)
            subsample    = trial.suggest_float("xgb_subsample", 0.6, 1.0)
            colsample    = trial.suggest_float("xgb_colsample", 0.6, 1.0)
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

# --------------------------------------------------------------------
# Haupt-Callable (Airflow Task)
# --------------------------------------------------------------------
def tune_and_train():
    try:
        # MLflow-Tracking
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5002"))
        mlflow.set_experiment("spam_tuning")  # **genau wie im UI**

        df = collect_training_data()
        X = df["text"].astype(str).to_numpy()
        y = df["label"].astype(int).to_numpy()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # ----- Optuna -----
        study = optuna.create_study(direction="maximize")
        with mlflow.start_run(run_name="optuna_search"):
            study.optimize(objective_factory(X_train, y_train), n_trials=40, show_progress_bar=False)
            mlflow.log_params(study.best_params)
            mlflow.log_metric("cv_f1_best", study.best_value)

        # Bestes Setup rekonstruieren
        p = study.best_params
        model_type = p["model_type"]
        ngram_hi   = p["ngram_hi"]
        max_feat   = p["max_features"]

        if model_type == "logreg":
            clf = LogisticRegression(C=p["logreg_C"], max_iter=400, solver="liblinear")
        elif model_type == "linear_svc":
            base = LinearSVC(C=p["svm_C"])
            # Kalibrieren, damit wir predict_proba haben
            clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
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

        pipe_best = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_hi), max_features=max_feat, min_df=2)),
            ("clf", clf),
        ])

        # ----- Final Fit + Registry/Promotion -----
        with mlflow.start_run(run_name="final_fit") as run:
            pipe_best.fit(X_train, y_train)

            # Val-Probas
            if hasattr(pipe_best.named_steps["clf"], "predict_proba"):
                val_prob = pipe_best.predict_proba(X_val)[:, 1]
            else:
                from sklearn.preprocessing import MinMaxScaler
                dec = pipe_best.decision_function(X_val)
                val_prob = MinMaxScaler().fit_transform(dec.reshape(-1, 1)).ravel()

            thr, p_thr, r_thr, f1_thr = pick_threshold(y_val, val_prob, target_precision=0.90)
            y_hat = (val_prob >= thr).astype(int)

            # Metriken
            f1   = f1_score(y_val, y_hat)
            prec = precision_score(y_val, y_hat, zero_division=0)
            rec  = recall_score(y_val, y_hat, zero_division=0)
            try:
                auc = roc_auc_score(y_val, val_prob)
            except Exception:
                auc = None

            # Logging
            mlflow.log_params({"model_type": model_type, "ngram_hi": ngram_hi, "max_features": max_feat})
            mlflow.log_metric("val_f1", f1)
            mlflow.log_metric("val_precision", prec)
            mlflow.log_metric("val_recall", rec)
            if auc is not None:
                mlflow.log_metric("val_roc_auc", auc)
            mlflow.log_metric("chosen_threshold", thr)

            # Lokale Sicherung (optional)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            with open(CURRENT_MODEL, "wb") as f:
                pickle.dump(pipe_best, f)

            # Threshold als Artefakt + lokal (solange Run offen ist!)
            _atomic_write_text(json.dumps({"threshold": float(thr)}, indent=2), THRESH_PATH)
            mlflow.log_artifact(str(THRESH_PATH), artifact_path="model")

            # In Registry registrieren (erstellt Model bei Bedarf automatisch)
            model_info = mlflow.sklearn.log_model(
                sk_model=pipe_best,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )

            # Model-Version finden, die zu diesem Run gehört
            client = MlflowClient()
            run_id = run.info.run_id
            mv_for_run = None
            for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
                if mv.run_id == run_id:
                    mv_for_run = mv
                    break
            if mv_for_run is None:
                raise RuntimeError("Konnte Model-Version zum aktuellen Run nicht finden.")
            new_version = mv_for_run.version

        # *** ab hier NACH dem Run-Context ***
        client = MlflowClient()

        # Auto-Promotion Policy (Stage-basierend, keine Aliases)
        new_metrics = client.get_run(run_id).data.metrics
        new_prec = new_metrics.get("val_precision", 0.0)
        new_f1   = new_metrics.get("val_f1", 0.0)

        current_prod = None
        latest_list = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if latest_list:
            current_prod = latest_list[0]

        if current_prod is None:
            better = True  # erstes Modell -> direkt Production
        else:
            prod_metrics = client.get_run(current_prod.run_id).data.metrics
            prod_f1 = prod_metrics.get("val_f1", 0.0)
            # Policy: Precision >= 0.90 und F1 nicht schlechter
            better = (new_prec >= 0.90) and (new_f1 >= prod_f1 - 1e-9)

        if better:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=new_version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"[registry] promoted v{new_version} → Production (val_f1={new_f1:.3f}, val_prec={new_prec:.3f})")
        else:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=new_version,
                stage="Staging",
            )
            print(f"[registry] kept v{new_version} in Staging (val_f1={new_f1:.3f}, val_prec={new_prec:.3f})")

        print(f"[train] model_type={model_type}")
        print(f"[train] wrote model → {CURRENT_MODEL}")
        print(f"[train] wrote threshold → {THRESH_PATH}")

    except Exception as e:
        # Deutlichere Fehlerausgabe im Airflow Log
        print("[ERROR] tune_and_train failed:\n" + "".join(traceback.format_exc()))
        # erneut auslösen, damit der Task als 'failed' markiert wird
        raise

# --------------------------------------------------------------------
# Airflow DAG
# --------------------------------------------------------------------
with DAG(
    dag_id="spam_train_tune",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,  # manuell
    catchup=False,
    tags=["spam", "train", "mlflow", "optuna"],
    description="Collect data, Optuna tune, log to MLflow Registry, save best model + threshold.",
) as dag:
    train = PythonOperator(task_id="tune_and_train", python_callable=tune_and_train)