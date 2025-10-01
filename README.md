# Spam Classifier — FastAPI & Airflow Batch Scoring

End-to-end spam classifier with a FastAPI service for real-time inference and an Apache Airflow (Docker + PostgreSQL + CeleryExecutor) DAG for batch scoring.

## Features
- Logistic Regression + TF-IDF (scikit-learn Pipeline)
- FastAPI `/predict` (JSON in, Prediction & Probability out)
- Airflow DAG for daily/manual batch scoring of CSVs
- Extended DAG: optional evaluation metrics after each batch
- Example model & example CSV included in repo

## Project Structure
```
mlops_spam/
├─ fastapi_app/             # FastAPI app
│  └─ main.py
├─ airflow_docker/          # Airflow subproject (Docker)
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
- Python 3.10–3.13 (local, for FastAPI)
- Docker & Docker Compose (for Airflow)

## Local API (FastAPI)
```bash
# venv (optional)
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn scikit-learn

# Start FastAPI (port 5000)
uvicorn fastapi_app.main:app --reload --port 5000
# -> http://127.0.0.1:5000/docs
```

**Example request**
```bash
curl -X POST http://127.0.0.1:5000/predict   -H "Content-Type: application/json"   -d '{"text":"Win a FREE prize now!!!"}'
```

## Batch Scoring with Airflow (Docker)
The metadata DB is PostgreSQL (backed by ./pgdata volume) + Redis broker.
```bash
cd airflow_docker
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up -d
# UI -> http://localhost:8080 (Admin login see logs or set manually)
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
- Inference threshold (`BEST_THRESHOLD`) configurable in DAG/FastAPI code.
- Airflow timezone set in `docker-compose.yml` (e.g. `Europe/Zurich`).
- Scheduler: adjust `schedule_interval` in DAG (e.g. `"0 2 * * *"`).

## Troubleshooting
- **Airflow UI not loading:** check logs `docker logs -f airflow-web`; change port if needed (`"8081:8080"`).
- **Login in Airflow:** reset password:  
  `docker exec -it airflow-web airflow users reset-password --username admin --password admin`
- **Missing packages:** extend `_PIP_ADDITIONAL_REQUIREMENTS` in `docker-compose.yml` (e.g. `nltk`).

---

# 🐳 Docker Cheat Sheet — Airflow & FastAPI Project

## 🚀 Workflow

1. **Start Docker Desktop**  
   - On macOS the whale icon in the menu bar → must be running.

2. **Go to project folder**  
   ```bash
   cd ~/Data\ Science/Projects/mlops_spam/airflow-docker
   ```

3. **Start Airflow stack (background)**  
   ```bash
   docker compose up -d
   ```
   - Starts Postgres, Redis, Airflow (Webserver, Scheduler, Worker).  
   - UI: [http://localhost:8080](http://localhost:8080)

4. **Stop containers**  
   ```bash
   docker compose down
   ```
   - Removes containers but keeps volumes.  
   - Remove everything (logs, DB):  
     ```bash
     docker compose down -v
     ```

---

## 🔑 Useful Commands

| Command | Description |
|---------|-------------|
| `docker ps` | Show running containers |
| `docker compose ps` | Show containers of this project |
| `docker logs -f airflow-web` | Follow logs of the Airflow webserver |
| `docker exec -it airflow-worker bash` | Open shell inside worker |
| `docker images` | List local images |
| `docker system df` | Check disk usage |
| `docker system prune` | Cleanup unused containers, networks, caches |

---

## ⚡ Resource Usage

- **Docker Desktop idle**: ~0.5–1.5 GB RAM, CPU minimal.  
- **Containers idle**: Airflow + DB + Redis → ~200–500 MB each.  
- **During DAG runs**: RAM/CPU usage depends on Python tasks (pandas, sklearn).  
- **Image builds**: short spikes of CPU/RAM.  

👉 In idle state: uncritical. Load only when running DAGs or builds.
