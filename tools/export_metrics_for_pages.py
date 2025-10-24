#!/usr/bin/env python3
"""
Exportiert Metriken aus Postgres (metrics_history) nach docs/metrics.json (+ CSV).

- DSN via METRICS_PG_DSN, Default: postgresql://airflow:airflow@127.0.0.1:5432/airflow_prod
- Erwartet Tabelle 'metrics_history' mit u.a.:
  ds, f1, precision, recall, roc_auc, threshold_used, suggested_threshold,
  n, tp, tn, fp, fn, spam_rate
"""
import os, json, csv, sys
import psycopg2

# Standard: auf der VM verbindet 127.0.0.1:5432 zum Postgres-Container (Port gemappt)
DSN = os.getenv(
    "METRICS_PG_DSN",
    "postgresql://airflow:airflow@127.0.0.1:5432/airflow_prod"
)

QUERY = """
SELECT
  ds,
  -- Zahlen casten wir auf double precision (JSON-freundlich)
  f1::double precision                         AS f1,
  precision::double precision                  AS precision,
  recall::double precision                     AS recall,
  roc_auc::double precision                    AS roc_auc,
  threshold_used::double precision             AS threshold_used,
  suggested_threshold::double precision        AS suggested_threshold,
  n, tp, tn, fp, fn,
  spam_rate::double precision                  AS spam_rate
FROM metrics_history
ORDER BY ds ASC
LIMIT 180;
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

    # JSON schreiben
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT_JSON} with {len(rows)} rows")

    # CSV schreiben (gleiche Spaltenreihenfolge)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} with {len(rows)} rows")

if __name__ == "__main__":
    main()
