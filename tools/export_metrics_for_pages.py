#!/usr/bin/env python3
"""
Exportiert Metriken aus Postgres (metrics_history) nach docs/metrics.json (+ CSV).
- DSN über env METRICS_PG_DSN, Default: postgresql://airflow:airflow@127.0.0.1:5432/mlflow_prod
- Erwartet Tabelle 'metrics_history' mit mind. ds, f1, precision, recall, threshold_used, suggested_threshold.
"""
import os, json, csv, sys
import psycopg2

DSN = os.getenv("METRICS_PG_DSN", "postgresql://airflow:airflow@127.0.0.1:5432/mlflow_prod")

QUERY = """
SELECT
  ds,
  f1,
  precision,
  recall,
  threshold_used,
  suggested_threshold,
  spam_rate,
  roc_auc,
  balanced_accuracy
FROM metrics_history
ORDER BY ds ASC;
"""

OUT_JSON = "docs/metrics.json"
OUT_CSV  = "docs/metrics.csv"

def main():
    try:
        with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
            cur.execute(QUERY)
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as e:
        print(f"[export] DB-Fehler: {e}", file=sys.stderr)
        sys.exit(2)

    # JSON
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT_JSON} with {len(rows)} rows")

    # CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} with {len(rows)} rows")

if __name__ == "__main__":
    main()
