# inspect_airflow_db.py
import sqlite3
import pandas as pd
from pathlib import Path

# Projektroot = Ordner über "scripts"
ROOT = Path(__file__).resolve().parents[1]

# ACHTUNG: Bindestrich-Ordnername ("airflow-docker"), nicht underscore
DB_PATH = (ROOT / "airflow-docker" / "db" / "airflow.db").resolve()

def read_sql(query: str, params: tuple = ()):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Airflow DB not found at: {DB_PATH.resolve()}")
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)

def list_dag_runs(dag_id: str | None = None, limit: int = 20) -> pd.DataFrame:
    base = """
    SELECT dag_id, run_id, state, start_date, end_date, run_type, external_trigger
    FROM dag_run
    {where}
    ORDER BY start_date DESC
    LIMIT ?
    """
    if dag_id:
        q = base.format(where="WHERE dag_id = ?")
        return read_sql(q, (dag_id, limit))
    else:
        q = base.format(where="")
        return read_sql(q, (limit,))

def list_task_instances(dag_id: str | None = None, run_id: str | None = None, limit: int = 50) -> pd.DataFrame:
    base = """
    SELECT dag_id, task_id, run_id, try_number, state, start_date, end_date, duration
    FROM task_instance
    {where}
    ORDER BY start_date DESC
    LIMIT ?
    """
    clauses, params = [], []
    if dag_id:
        clauses.append("dag_id = ?")
        params.append(dag_id)
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    q = base.format(where=where)
    return read_sql(q, (*params, limit))

def last_xcoms(dag_id: str | None = None, limit: int = 20) -> pd.DataFrame:
    # XCom enthält kleine Payloads (JSON/BLOB); für Überblick nur Metadaten anzeigen
    base = """
    SELECT dag_id, task_id, run_id, key, timestamp
    FROM xcom
    {where}
    ORDER BY timestamp DESC
    LIMIT ?
    """
    if dag_id:
        q = base.format(where="WHERE dag_id = ?")
        return read_sql(q, (dag_id, limit))
    else:
        q = base.format(where="")
        return read_sql(q, (limit,))

if __name__ == "__main__":
    # Beispiele:
    print("\n=== Recent DAG runs ===")
    print(list_dag_runs(limit=10))

    print("\n=== Recent DAG runs for 'spam_batch_scoring' ===")
    print(list_dag_runs(dag_id="spam_batch_scoring", limit=10))

    print("\n=== Recent task instances (all DAGs) ===")
    print(list_task_instances(limit=10))

    print("\n=== Recent task instances for 'spam_batch_scoring' ===")
    print(list_task_instances(dag_id="spam_batch_scoring", limit=10))

    print("\n=== Recent XComs (if any) ===")
    print(last_xcoms(limit=10))