# Spam Classifier — Flask API & Airflow Batch Scoring

End-to-end spam classifier with a Flask API for real-time inference and an Apache Airflow (Docker + PostgreSQL + CeleryExecutor) DAG for scalable batch scoring.

## Features
- Logistic Regression + TF-IDF (scikit-learn Pipeline)
- Flask API `/predict` (JSON in, Prediction & Probability out)
- Airflow DAG for daily/manual batch scoring of CSVs
- Runs with **CeleryExecutor** (Redis broker + scalable workers)
- Example model & example CSV included in repo

## Project Structure
```
mlops_spam/
├─ app/                      # Flask app (optional)
├─ airflow_docker/           # Airflow subproject (Docker)
│  ├─ dags/
│  │  └─ spam_batch_scoring_dag.py
│  ├─ data/
│  │  └─ messages.csv
│  ├─ models/
│  │  └─ logreg_spam_pipeline.pkl
│  └─ docker-compose.yml
├─ models/
│  └─ logreg_spam_pipeline.pkl
├─ data/
│  └─ messages.csv
└─ README.md
```

## Requirements
- Python 3.10–3.13 (local, for Flask only)
- Docker & Docker Compose (for Airflow)

## Local API (Flask)
```bash
# venv (optional)
python -m venv .venv && source .venv/bin/activate
pip install flask scikit-learn

# Start (e.g. app.py)
python app.py
# -> http://127.0.0.1:5000
```

**Example request**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Win a FREE prize now!!!"}'
```

## Batch Scoring with Airflow (Docker + CeleryExecutor)
The metadata DB is PostgreSQL (backed by ./pgdata volume). Redis is used as the Celery broker.

```bash
cd airflow_docker
echo "AIRFLOW_UID=$(id -u)" > .env

# One-time init (DB migrate, admin user creation)
docker compose run --rm airflow-init

# Start webserver, scheduler & worker
docker compose up -d airflow-webserver airflow-scheduler airflow-worker redis postgres

# UI -> http://localhost:8080 (Login: admin / admin)
```

### Scaling workers
You can run multiple workers for parallel tasks:
```bash
docker compose up -d --scale airflow-worker=3
```

## .env example
```
AIRFLOW_UID=1000
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

AIRFLOW__CORE__EXECUTOR=CeleryExecutor
AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://airflow:airflow@postgres/airflow

AIRFLOW__CORE__PARALLELISM=16
AIRFLOW__CORE__DAG_CONCURRENCY=16
AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG=1

AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__WEBSERVER__DEFAULT_UI_TIMEZONE=Europe/Zurich
```

**Trigger DAG**
- In UI enable `spam_batch_scoring` → ▶️ Trigger
- Output: `airflow_docker/data/predictions_YYYY-MM-DD.csv`

**CLI (inside container)**
```bash
docker exec -it airflow-web airflow dags trigger spam_batch_scoring
docker exec -it airflow-web airflow dags list-runs -d spam_batch_scoring
```

## Data & Model
- Example CSV: `data/messages.csv` (header: `id,text`)
- Model: `models/logreg_spam_pipeline.pkl` (Pipeline with `TfidfVectorizer` + `LogisticRegression`)
- For Airflow both are mounted into container (`/opt/airflow/data`, `/opt/airflow/models`).

## Configuration
- Inference threshold (`BEST_THRESHOLD`) configurable in DAG/Flask code.
- Airflow timezone set in `docker-compose.yml` (e.g. `Europe/Zurich`).
- Scheduler: adjust `schedule_interval` in DAG (e.g. `"0 2 * * *"`).

## Troubleshooting
- **Airflow UI not loading:** check logs  
  ```bash
  docker logs -f airflow-web
  ```
  Change port in `docker-compose.yml` if needed (e.g. `"8081:8080"`).
- **Reset password:**  
  ```bash
  docker exec -it airflow-web airflow users reset-password --username admin --password admin
  ```
- **Missing packages:** extend `_PIP_ADDITIONAL_REQUIREMENTS` in `docker-compose.yml` (e.g. `nltk`).
- **No workers visible:** check `docker logs -f airflow-worker` and ensure Redis is running.

## License
MIT License