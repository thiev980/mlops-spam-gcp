# Spam Model Monitoring & Live Inference  
**Airflow + MLflow + Postgres + Grafana + FastAPI + Cloudflare + GitHub Actions/Pages**

[![Daily Metrics Update](https://github.com/thiev980/mlops-spam-gcp/actions/workflows/update-metrics.yml/badge.svg)](https://github.com/thiev980/mlops-spam-gcp/actions/workflows/update-metrics.yml)
[![GitHub Pages](https://img.shields.io/badge/live%20metrics-online-success)](https://thiev980.github.io/mlops-spam-gcp/)

End-to-end MLOps for a spam detection system: **daily batch evaluation**, **model registry**, **dashboards**, and **live inference** via FastAPI — securely published through **Cloudflare Tunnel** and embedded into **GitHub Pages**.

---

## 🔭 Overview

- **Airflow** orchestrates daily jobs (ETL, evaluation, export).
- **MLflow** logs runs & models; optional promotion policy via Registry.
- **PostgreSQL** stores metrics and historical runs.
- **Grafana** visualizes trends and drifts from Postgres.
- **GitHub Actions** fetches daily `metrics.json` from the VM and deploys it to **GitHub Pages**.
- **FastAPI** serves the current production model via `/predict`.
- **Cloudflare Tunnel** publishes the API securely as an HTTPS subdomain.
- **Docker Compose** manages everything reproducibly on the GCP VM.

---

## 🧭 Architecture (Services & Data Flow)

```mermaid
flowchart LR
  subgraph GCP_VM["GCP VM (Docker Compose)"]
    A[Airflow\n(web/scheduler/worker)]
    M[MLflow Server]
    DB[(PostgreSQL)]
    R[Redis]
    F[FastAPI\nModel Serving]
    Gr[Grafana]
  end

  subgraph GitHub["GitHub"]
    GA[GitHub Actions]
    GP[GitHub Pages\n(Live Dashboard)]
  end

  %% Batch Flow
  A -- write metrics --> DB
  A -- log runs/models --> M
  M -- backend store --> DB

  %% Dashboard Export
  GA -- scp/ssh fetch --> GCP_VM
  GA -- publish metrics.json --> GP

  %% Observability
  Gr -- queries --> DB

  %% Live Inference
  CF[Cloudflare Tunnel\nfastapi.thiev980.com] --- F
  Browser[User Browser] -- fetch('/predict') --> CF
  Browser --> GP
```

---

## 🔄 Daily Workflow (Batch)

```mermaid
graph LR
  A[Airflow DAGs\n(spam_daily_orchestration)] --> H[metrics_history in Postgres]
  H --> X[Export Script\n(export -> metrics.json)]
  X -->|SSH/SCPO| GA[GitHub Actions]
  GA -->|commit to gh-pages| GP[GitHub Pages Dashboard]
```

**Every day at 03:05 UTC:**
1. Airflow evaluates the current production model and writes metrics to **Postgres**  
2. The export script generates **`metrics.json`**  
3. **GitHub Actions** fetches it via SSH and deploys it → **GitHub Pages**

👉 Live Dashboard: **https://thiev980.github.io/mlops-spam-gcp/**

---

## ⚡ Live Inference (FastAPI)

- Public domain via Cloudflare Tunnel: **`https://fastapi.thiev980.com`**
- Healthcheck: `GET /` → `{"status":"ok", "model":"model_current.pkl", "threshold":...}`
- Inference: `POST /predict` with `{"text": "..."}`

**Example:**
```bash
curl -s -X POST "https://fastapi.thiev980.com/predict"   -H "Content-Type: application/json"   -d '{"text":"Win a FREE iPhone now!"}'
```

The GitHub Page includes a small **live test section** at the bottom (directly calling FastAPI).

---

## 🧩 Components

- **Airflow**: DAGs `spam_daily_orchestration`, `spam_train_tune`  
  - Handles training/tuning with Optuna, logs to MLflow, and applies a promotion policy  
- **MLflow**: Tracking server with Postgres backend store and local artifact storage  
- **Postgres**: Stores metric history and MLflow backend metadata  
- **Grafana**: Dashboards querying Postgres (optionally system metrics later)  
- **FastAPI**: Serves production pipeline (`model_current.pkl`) + threshold with CORS for GitHub Pages  
- **Cloudflare Tunnel**: Secure HTTPS ingress without open ports  
- **GitHub Actions/Pages**: CI export & live metric hosting

---

## 🛠️ Runbook (Quick Start)

**Docker Compose (Start/Stop)**
```bash
# From project root
docker compose -f docker-compose.prod.yml up -d        # start all
docker compose -f docker-compose.prod.yml down         # stop all
docker compose -f docker-compose.prod.yml ps           # check status
```

**Airflow – Initial DB Setup**
```bash
docker compose -f docker-compose.prod.yml up airflow-init
```

**Targeted Services**
```bash
docker compose -f docker-compose.prod.yml up -d postgres redis
docker compose -f docker-compose.prod.yml up -d mlflow grafana
docker compose -f docker-compose.prod.yml up -d airflow-web airflow-scheduler airflow-worker
docker compose -f docker-compose.prod.yml up -d fastapi
```

**Logs**
```bash
docker logs -f airflow-web
docker logs -f airflow-worker
docker logs -f mlflow
docker logs -f spam-fastapi
```

---

## 🔐 Cloudflare Tunnel (FastAPI)

- Named Tunnel `spam-fastapi` → **`fastapi.thiev980.com`**
- Managed by systemd:
```bash
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```
- Verify connection:
```bash
cloudflared tunnel list
curl -s https://fastapi.thiev980.com/
```

(Compose services remain internal; the tunnel handles HTTPS ingress securely.)

---

## 🔧 Local Ports

| Service      | Port (Host) |
|--------------|-------------|
| Airflow Web  | 8080        |
| MLflow       | 5002        |
| FastAPI      | 5000 (internal; public via Cloudflare) |
| Grafana      | 3000        |
| Postgres     | 5432        |
| Redis        | 6379        |

> No firewall rules or open ports needed — Cloudflare provides TLS and reverse proxying.

---

## 🧪 Model Training & Promotion

- **Manual Training:** Trigger in Airflow UI → `spam_train_tune`
- Threshold Selection: prioritizes **Precision ≥ 0.90**, otherwise slightly relaxed; fallback = best F1
- Promotion Policy: New model promoted to *Production* if  
  - `Precision ≥ 0.90`
  - `F1 ≥ current Production F1`
  - `Recall` not more than **2 percentage points worse**

> For fair comparisons, a fixed rolling evaluation window (e.g., last 7–14 days) is recommended.

---

## 🧰 Troubleshooting

- **Airflow “SQLAlchemy URL […] ''”** → missing ENV vars. Ensure Compose defines:
  - `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`
  - `AIRFLOW__CORE__EXECUTOR`
  - `AIRFLOW__CELERY__*`
- **FastAPI not reachable:** check container logs and model path (`/opt/airflow/models`).
- **GitHub Pages not updating:** check `update-metrics.yml` logs and SSH key validity.

---

## 📎 Links

- Dashboard → **https://thiev980.github.io/mlops-spam-gcp/**
- Live API → **https://fastapi.thiev980.com** (`/predict`)

---

## 👤 Contact

**Thierry Figini**  
🔗 [GitHub](https://github.com/thiev980) • [LinkedIn](https://www.linkedin.com/in/thierryfigini/)  
✉️ thierry_figini@me.com
