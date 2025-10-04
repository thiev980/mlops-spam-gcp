# spam_daily_orchestration.py
from __future__ import annotations
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 9, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

# Hinweis:
# - Wir geben ds explizit als conf weiter. Deine Sub-DAGs nutzen bereits "ds",
#   manche Tasks können optional dieses conf["ds"] lesen, falls nötig.
# - wait_for_completion=True blockiert bis die getriggerte DAG (für *diesen* Run)
#   fertig ist (oder fehlschlägt).
with DAG(
    dag_id="spam_daily_orchestration",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",   # täglich 02:00
    catchup=False,
    tags=["spam", "orchestration"],
    description="Trigger sequence: LLM gen -> model score -> eval (waits between steps).",
) as dag:

    # 1) LLM-Generierung starten und auf Erfolg warten
    gen = TriggerDagRunOperator(
        task_id="trigger_spam_gen_llm",
        trigger_dag_id="spam_gen_llm",
        conf={"ds": "{{ ds }}"},
        reset_dag_run=True,
        wait_for_completion=True,
        poke_interval=15,                # alle 15s nachschauen
        allowed_states=["success"],
        failed_states=["failed"],
    )

    # 2) Scoring starten (nutzt die von 1) erzeugten Dateien)
    score = TriggerDagRunOperator(
        task_id="trigger_spam_score_model",
        trigger_dag_id="spam_score_model",
        conf={"ds": "{{ ds }}"},
        reset_dag_run=True,
        wait_for_completion=True,
        poke_interval=15,
        allowed_states=["success"],
        failed_states=["failed"],
    )

    # 3) Evaluation + Plot + Upsert
    evaluate = TriggerDagRunOperator(
        task_id="trigger_spam_eval",
        trigger_dag_id="spam_eval",
        conf={"ds": "{{ ds }}"},
        reset_dag_run=True,
        wait_for_completion=True,
        poke_interval=15,
        allowed_states=["success"],
        failed_states=["failed"],
    )

    gen >> score >> evaluate