# spam_daily_orchestration.py
from __future__ import annotations
from datetime import timedelta
import pendulum

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": pendulum.datetime(2025, 9, 1, tz="Europe/Zurich"),
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="spam_daily_orchestration",
    schedule="0 2 * * *",       # täglich 02:00
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["spam", "orchestration"],
    description="Trigger sequence: LLM gen -> model score -> eval (waits between steps).",
) as dag:

    gen = TriggerDagRunOperator(
        task_id="trigger_spam_gen_llm",
        trigger_dag_id="spam_gen_llm",
        conf={"ds": "{{ ds }}"},
        reset_dag_run=True,
        wait_for_completion=True,
        poke_interval=15,
        allowed_states=["success"],
        failed_states=["failed"],
    )

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
