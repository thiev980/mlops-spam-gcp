# spam_gen_llm.py
from __future__ import annotations
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import os, json, random, requests, pandas as pd

DATA_DIR   = Path("/opt/airflow/data")
INCOMING   = DATA_DIR / "incoming"
LABELS     = DATA_DIR / "labels"
INCOMING.mkdir(parents=True, exist_ok=True)
LABELS.mkdir(parents=True, exist_ok=True)

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 9, 1),
    "retries": 0,                 # wichtig: keine automatischen Retries beim teuren LLM-Call
    "retry_delay": timedelta(minutes=1),
}

def _atomic_write(df: pd.DataFrame, dest: Path):
    """Write CSV atomically to avoid half-written files on failures."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)

# 2) Robuste, gechunkte LLM-Generierung
def _llm_generate(n: int, spam_pct: int = 10) -> list[dict]:
    import os, json, time, math, requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    endpoint = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    api_key  = os.getenv("OPENAI_API_KEY")
    model    = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    def _session():
        s = requests.Session()
        r = Retry(
            total=3, backoff_factor=1.5,
            status_forcelist=[429,500,502,503,504],
            allowed_methods=["POST"], raise_on_status=False
        )
        a = HTTPAdapter(max_retries=r)
        s.mount("https://", a); s.mount("http://", a)
        return s

    def _one_batch(batch_size: int, start_id: int) -> list[dict]:
        prompt = f"""
Return ONLY a JSON array (no markdown).
Each item: {{"id": number, "text": string, "label": 0 or 1}} (1=spam,0=ham).
Generate {batch_size} email-like messages (5–20 words), ~{spam_pct}% spam.
IDs must be consecutive starting at {start_id}.
Example format:
[{{"id":{start_id},"text":"Win a prize now","label":1}},{{"id":{start_id+1},"text":"Team meeting moved to 10am","label":0}}]
""".strip()

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,  # weniger "kreativ" -> hält sich besser ans Schema
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "batch_messages",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "text": {"type": "string"},
                                        "label": {"type": "integer", "enum": [0, 1]}
                                    },
                                    "required": ["id", "text", "label"],
                                    "additionalProperties": False
                                },
                                "minItems": batch_size,
                                # optional: "maxItems": batch_size
                            }
                        },
                        "required": ["items"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        }

        sess = _session()
        t0 = time.time()
        # Per-Batch Timeout: (connect=10s, read=90s)
        resp = sess.post(endpoint, headers=headers, json=body, timeout=(10, 90))
        dt = time.time() - t0
        if resp.status_code != 200:
            raise RuntimeError(f"[batch {start_id}-{start_id+batch_size-1}] OpenAI HTTP {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        text = data["choices"][0]["message"]["content"]

        # Primär: direkt JSON parsen und "items" entnehmen
        try:
            obj = json.loads(text)
            items = obj.get("items", None)
        except Exception:
            items = None

        # Fallbacks: falls doch mal kein reines JSON kommt
        if not items:
            # 1) Versuche, den JSON-Block zwischen dem ersten "{" und letzten "}" zu extrahieren
            import re
            lb = text.find("{")
            rb = text.rfind("}")
            if lb != -1 and rb != -1 and rb > lb:
                frag = text[lb:rb+1]
                try:
                    obj = json.loads(frag)
                    items = obj.get("items", None)
                except Exception:
                    items = None

        if not items or not isinstance(items, list) or len(items) == 0:
            preview = (text[:600] + "...") if len(text) > 600 else text
            raise ValueError(f"[batch {start_id}] Empty/invalid JSON (no 'items' array). Preview:\n{preview}")

        # Normalisieren/validieren
        out = []
        for x in items:
            if isinstance(x, dict) and "id" in x and "text" in x and "label" in x:
                lab = 1 if int(x["label"]) == 1 else 0
                out.append({"id": int(x["id"]), "text": str(x["text"]), "label": lab})

        if len(out) == 0:
            raise ValueError(f"[batch {start_id}] Parsed JSON but no valid objects found.")
        return out

    # sinnvolle Batchgröße (50–80 funktioniert meist gut)
    batch = 50
    num_batches = math.ceil(n / batch)
    results: list[dict] = []
    next_id = 1
    for i in range(num_batches):
        want = batch if (i < num_batches - 1) else (n - batch*(num_batches-1))
        # Retry-Schleife *pro Batch* (z. B. 2 Versuche bei harten Read-Timeouts)
        for attempt in range(2):
            try:
                part = _one_batch(want, next_id)
                results.extend(part)
                next_id += want
                break
            except Exception as e:
                print(f"[batch {next_id}] attempt {attempt+1} failed: {e}")
                if attempt == 0:
                    time.sleep(3.0)
                else:
                    raise
        # kleine Pause zwischen Batches hilft manchmal
        time.sleep(0.5)
    if not results:
        raise ValueError("LLM returned no items across all batches")
    return results

def generate_llm_batches(ds: str, **_):
    # Ensure dirs exist (idempotent)
    INCOMING.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    incoming_path = INCOMING / f"{ds}.csv"
    labels_path   = LABELS / f"{ds}.csv"

    # Idempotenz: wenn schon vorhanden, NICHT neu generieren
    if incoming_path.exists() and labels_path.exists():
        print(f"{incoming_path} and {labels_path} exist, skipping generation.")
        return

    rng = random.Random(hash(ds) % 2_000_000)
    n = rng.randint(120, 180)  # tägliche Variation
    rows = _llm_generate(n=n, spam_pct=10)
    df = pd.DataFrame(rows)

    # Eingangs-Datei (ohne Labels)
    _atomic_write(df[["id","text"]], incoming_path)
    # Labels separat
    _atomic_write(df[["id","label"]], labels_path)
    print(f"Wrote {len(df)} rows → {incoming_path} & {labels_path}")

with DAG(
    dag_id="spam_gen_llm",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",   # 02:00 täglich
    catchup=False,
    tags=["spam","llm","generate"],
    description="Generate unlabeled messages + separate labels via LLM",
) as dag:
    gen = PythonOperator(task_id="generate_llm_batches", python_callable=generate_llm_batches)