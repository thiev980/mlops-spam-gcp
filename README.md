# Spam Model Monitoring (Airflow + MLflow + Grafana + GitHub Actions)

[![Daily Metrics Update](https://github.com/thiev980/mlops-spam-gcp/actions/workflows/update-metrics.yml/badge.svg)](https://github.com/thiev980/mlops-spam-gcp/actions/workflows/update-metrics.yml)
[![GitHub Pages](https://img.shields.io/badge/live%20metrics-online-success)](https://thiev980.github.io/mlops-spam-gcp/)

An end-to-end MLOps project for **daily monitoring of a spam detection model**.
Automatically orchestrated, visualized, and published — entirely in the cloud.

---

## Architecture

- **Airflow** – orchestrates daily evaluation runs and stores metrics in PostgreSQL 
- **MLflow** – manages models and threshold artifacts
- **Grafana** – visualizes historical trends and drifts 
- **GitHub Actions** – exports daily metrics from the VM 
- **GitHub Pages** – hosts the live performance preview (JSON + chart)

---

## Workflow

```mermaid
graph LR
    A[Airflow DAGs] --> B[PostgreSQL metrics_history]
    B --> C[export_metrics_for_pages.py]
    C --> D[GitHub Actions]
    D --> E[gh-pages branch]
    E --> F[GitHub Pages Dashboard]
```

Every night at **03:05 UTC**:
1. Airflow writes the latest model metrics to Postgres 
2. `export_metrics_for_pages.py` exports them as JSON
3. GitHub Actions fetches the file from the VM 
4. The `metrics.json` is automatically deployed → [Live Dashboard](https://thiev980.github.io/mlops-spam-gcp/)

---

## Live Monitoring

**[View the latest deployment](https://thiev980.github.io/mlops-spam-gcp/)**  
*(auto-updated daily via CI/CD)*

![Metrics Trend Screenshot](docs/assets/metrics_trend.png)

---

## Stack

| Category | Tool / Framework |
|------------|------------------|
| Orchestration | Apache Airflow |
| Tracking | MLflow |
| Monitoring | Grafana |
| Database | PostgreSQL |
| Automation | GitHub Actions |
| Hosting | GitHub Pages |

---

## Contact

👤 **Thierry Figini**  
🔗 [GitHub-Profil](https://github.com/thiev980) • [LinkedIn](https://www.linkedin.com/in/thierryfigini/)  
✉️ thierry_figini@me.com
