# End-to-end spam classifier with a Flask API for real-time inference and an Apache Airflow (Docker + PostgreSQL) DAG for batch scoring.

## Features
- Logistic Regression + TF-IDF (scikit-learn Pipeline)
- Flask API `/predict` (JSON in, Prediction & Probability out)
- Airflow DAG for daily/manual batch scoring of CSVs
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

## Batch Scoring with Airflow (Docker)
The metadata DB is PostgreSQL (backed by ./pgdata volume).

```bash
cd airflow_docker
echo "AIRFLOW_UID=$(id -u)" > .env

# One-time init (DB migrate, admin user creation)
docker compose run --rm airflow-init

# Start webserver & scheduler
docker compose up -d airflow-webserver airflow-scheduler

# UI -> http://localhost:8080 (Login: admin / admin)
```

## .env example
```
AIRFLOW_UID=1000
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow
AIRFLOW__CORE__EXECUTOR=SequentialExecutor
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
- **Airflow UI not loading:** check logs `docker logs -f airflow-web`; change port if needed (`"8081:8080"`).
- **Reset password:**  
  ```bash
  docker exec -it airflow-web airflow users reset-password --username admin --password admin
  ```
- **Missing packages:** extend `_PIP_ADDITIONAL_REQUIREMENTS` in `docker-compose.yml` (e.g. `nltk`).

## License
Add your license here (e.g. MIT).
"""

Path("README.md").write_text(readme_content, encoding="utf-8")