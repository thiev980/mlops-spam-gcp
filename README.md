# MLOps Spam Detection Pipeline

End-to-end MLOps pipeline for SMS spam classification, built with **Airflow**, **MLflow**, **PostgreSQL**, **Redis**, and **Grafana** — containerized via **Docker Compose** and ready for deployment on **Google Cloud**.

---

## Overview

This project demonstrates a complete MLOps workflow for a text classification use case (spam detection).  
It includes data ingestion, preprocessing, model training with hyperparameter tuning, experiment tracking, and performance monitoring — all orchestrated with Airflow and logged in MLflow.

---

## Tech Stack

| Component | Purpose |
|------------|----------|
| **Airflow** | Workflow orchestration (training, evaluation, retraining) |
| **MLflow** | Experiment tracking & model registry |
| **PostgreSQL** | Metadata storage (Airflow + MLflow) |
| **Redis** | Message broker for CeleryExecutor |
| **Grafana** | Metrics visualization from MLflow DB |
| **Docker Compose** | Containerized environment (Dev/Prod) |

---

## Project Structure

```
mlops-spam-gcp/
├── airflow-docker/
│   ├── dags/                # Airflow DAGs (pipelines)
│   ├── models-dev/          # Saved models (dev)
│   ├── data-dev/            # Local data inputs
│   ├── logs-dev/            # Airflow logs
│   ├── Dockerfile           # Airflow base image
│   ├── Dockerfile.mlflow    # MLflow service image
│   └── requirements.txt     # Python dependencies for Airflow
├── fastapi_app/             # Optional: REST API for inference
├── notebooks/               # Exploratory notebooks
├── docker-compose.dev.yml   # Dev environment
├── docker-compose.prod.yml  # Prod environment
├── .env.dev / .env.prod     # Environment variables
└── Makefile                 # Quick commands for switching environments
```

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/thiev980/mlops-spam-gcp.git
cd mlops-spam-gcp
```

### 2. Spin up the **Dev** environment
```bash
make up-dev
```
Services will start:
- Airflow → [http://localhost:8080](http://localhost:8080)  
- MLflow → [http://localhost:5002](http://localhost:5002)  
- Grafana → [http://localhost:3000](http://localhost:3000)

### 3. Trigger a pipeline
In the Airflow UI, trigger the DAG **`spam_train_tune`** to start model training and log metrics to MLflow.

### 4. Visualize results
In Grafana, connect to the MLflow Postgres DB and visualize metrics (F1, precision, recall, ROC AUC, etc.).

---

## Deployment

The same stack can be deployed to **Google Cloud** using:
- **Cloud Run** for the services (Airflow, MLflow, FastAPI)
- **Cloud SQL** for PostgreSQL
- **Cloud Storage** for artifacts and data
- **Cloud Logging / Monitoring** for observability

---

## Key Features

- Automated training & retraining via Airflow DAGs  
- Centralized experiment tracking in MLflow  
- Real-time monitoring dashboards in Grafana  
- Isolated Dev & Prod Docker Compose environments  
- Cloud-ready architecture for GCP deployment  

---

## Author

**Thi Fi**  
Editorial Data Analyst & MLOps Enthusiast  
📍 Zürich  

---

## License

This project is released under the MIT License. See `LICENSE` for details.
