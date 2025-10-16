# dags/spam_gen_llm.py
from __future__ import annotations

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import os, json, time, math, random, re, unicodedata

import pandas as pd
import requests

# --------------------------------------------------------------------
# Pfade & Konstanten
# --------------------------------------------------------------------
DATA_DIR   = Path("/opt/airflow/data")
INCOMING   = DATA_DIR / "incoming"
LABELS     = DATA_DIR / "labels"
PREDS      = DATA_DIR / "predictions"
INCOMING.mkdir(parents=True, exist_ok=True)
LABELS.mkdir(parents=True, exist_ok=True)

DEFAULT_ARGS = {
    "owner": "ml_engineer",
    "start_date": datetime(2025, 9, 1),
    "retries": 0,                  # kein Auto-Retry bei LLM
    "retry_delay": timedelta(minutes=1),
}

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _atomic_write(df: pd.DataFrame, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)

def _seed_from_ds(ds: str) -> int:
    # stabiler, deterministischer Seed aus Datum (YYYYMMDD)
    return int(pd.to_datetime(ds).strftime("%Y%m%d"))

# ---------- Adversarial Seeds (FN/FP von gestern) ----------
def _load_adversarial_seeds(ds: str, k_each: int = 12) -> list[str]:
    """
    Nimmt die Predictions & Labels von VORTAG (falls vorhanden) und
    extrahiert ein paar 'hard cases':
      - FN (spam==1, pred==0)
      - FP (spam==0, pred==1)
    Gibt eine kleine Liste von Beispiel-Texten zurück (ohne Duplikate).
    """
    try:
        prev = (pd.to_datetime(ds) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        return []

    preds_path  = PREDS  / f"{prev}.csv"
    labels_path = LABELS / f"{prev}.csv"
    if not preds_path.exists() or not labels_path.exists():
        return []

    try:
        df_p = pd.read_csv(preds_path)
        df_l = pd.read_csv(labels_path)
        if "prediction" in df_p.columns:
            df_p = df_p.rename(columns={"prediction": "pred"})
        if "label" in df_l.columns:
            df_l = df_l.rename(columns={"label": "label_gt"})

        df = df_p.merge(df_l, on="id", how="inner")
        text_col = "text" if "text" in df.columns else None
        if text_col is None:
            return []

        fn = df[(df["label_gt"] == 1) & (df["pred"] == 0)][text_col].dropna().tolist()
        fp = df[(df["label_gt"] == 0) & (df["pred"] == 1)][text_col].dropna().tolist()

        rng = random.Random(_seed_from_ds(ds))
        samples = []
        if fn:
            samples += rng.sample(fn, k=min(k_each, len(fn)))
        if fp:
            samples += rng.sample(fp, k=min(k_each, len(fp)))

        seen, out = set(), []
        for t in samples:
            t = str(t).strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t[:220])  # prompt kompakt halten
        return out[: 2 * k_each]
    except Exception:
        return []

# ---------- Hardening & Shuffle ----------
def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

HOMO = str.maketrans({"o": "ο", "a": "ɑ", "e": "е", "i": "і", "l": "ⅼ", "c": "ϲ"})

def _obfuscate_text(s: str, rng: random.Random) -> str:
    choice = rng.choice([0, 1, 2, 3])
    if choice == 0:
        return s.translate(HOMO)
    if choice == 1:
        return re.sub(r"\.", "[.]", s)
    if choice == 2:
        return s.replace(".", "\u200b.\u200b").replace("/", "\u200b/\u200b")
    if choice == 3:
        return re.sub(r"a", "4", s)
    return s

def _break_alternation(labels: list[int]) -> bool:
    runs = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    return runs >= len(labels) - 2

def _harden_and_shuffle(rows: list[dict], rng: random.Random, spam_target=0.15, tol=0.05):
    # dedup nach Normalisierung
    seen, uniq = set(), []
    for r in rows:
        key = _normalize(r["text"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    rows = uniq

    # grobe Spamquote (Simulation)
    n = len(rows)
    lo, hi = (spam_target - tol) * n, (spam_target + tol) * n
    spam_idx = [i for i, r in enumerate(rows) if r["label"] == 1]
    if len(spam_idx) < lo:
        need = int(lo - len(spam_idx))
        ham_idx = [i for i, r in enumerate(rows) if r["label"] == 0]
        for i in rng.sample(ham_idx, min(need, len(ham_idx))):
            rows[i]["label"] = 1
            rows[i]["text"] = "[PROMO] " + rows[i]["text"]
    elif len(spam_idx) > hi:
        drop = int(len(spam_idx) - hi)
        for i in rng.sample(spam_idx, min(drop, len(spam_idx))):
            rows[i]["label"] = 0
            rows[i]["text"] = rows[i]["text"].replace("WIN", "Thanks")

    # 30% der Spam-Texte obfuskieren
    spam_idx = [i for i, r in enumerate(rows) if r["label"] == 1]
    if spam_idx:
        for i in rng.sample(spam_idx, k=max(1, int(0.3 * len(spam_idx)))):
            rows[i]["text"] = _obfuscate_text(rows[i]["text"], rng)

    # mehrfach mischen, bis Alternation weg ist
    for _ in range(4):
        rng.shuffle(rows)
        if not _break_alternation([r["label"] for r in rows]):
            break
    return rows

# ---------- LLM-Generierung (mit Seeds) ----------
def _llm_generate(n: int, ds: str, spam_pct: int = 15) -> list[dict]:
    endpoint = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    api_key  = os.getenv("OPENAI_API_KEY")
    model    = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    def _session():
        s = requests.Session()
        r = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False,
        )
        a = HTTPAdapter(max_retries=r)
        s.mount("https://", a)
        s.mount("http://", a)
        return s

    # adversarial seeds aus dem Vortag
    seeds = _load_adversarial_seeds(ds, k_each=10)
    seeds_blurb = ""
    if seeds:
        sample = "\n".join(f"- {t}" for t in seeds[:12])
        seeds_blurb = (
            "\nUse the following example styles as inspiration (do NOT copy verbatim):\n"
            f"{sample}\n"
        )

    def _one_batch(batch_size: int, start_id: int) -> list[dict]:
        prompt = f"""
Return ONLY a JSON object with field "items": an array of objects.
Each item: {{"id": number, "text": string, "label": 0 or 1}} where 1=spam, 0=ham.

Generate {batch_size} diverse short messages (5–35 words) across topics.
Target spam share ~{spam_pct}% (allow ±5%). Do NOT alternate labels or follow patterns.
Randomize order and avoid near-duplicates.

Spam variants: sweepstakes, phishing/login, parcel/customs, invoice fraud, crypto get-rich,
romance, fake support, gift cards. Obfuscate a subset: leetspeak, homoglyphs, zero-width spaces,
masked URLs (example[.]com), spaced phone numbers.

Ham variants: work updates, meeting notes, friendly chats, delivery confirmations,
legit newsletters/promos (borderline but label 0), school/sports announcements.

Style variety: some formal, some casual; add typos occasionally; mix emoji; different lengths.
Language: primarily English, allow some German/Swiss-German.

IDs must be consecutive starting at {start_id}.
{seeds_blurb}
Example structure:
{{"items":[{{"id":{start_id},"text":"Team moved standup to 10:30","label":0}}]}}
""".strip()

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
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
                                        "label": {"type": "integer", "enum": [0, 1]},
                                    },
                                    "required": ["id", "text", "label"],
                                    "additionalProperties": False,
                                },
                                "minItems": batch_size,
                            }
                        },
                        "required": ["items"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        }

        sess = _session()
        resp = sess.post(endpoint, headers=headers, json=body, timeout=(10, 90))
        if resp.status_code != 200:
            raise RuntimeError(
                f"[batch {start_id}-{start_id+batch_size-1}] HTTP {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
        text = data["choices"][0]["message"]["content"]

        # JSON parse (robust)
        try:
            obj = json.loads(text)
            items = obj.get("items", None)
        except Exception:
            items = None

        if not items:
            lb, rb = text.find("{"), text.rfind("}")
            if lb != -1 and rb != -1 and rb > lb:
                try:
                    obj = json.loads(text[lb : rb + 1])
                    items = obj.get("items", None)
                except Exception:
                    items = None

        if not items or not isinstance(items, list):
            preview = (text[:600] + "...") if len(text) > 600 else text
            raise ValueError(f"[batch {start_id}] invalid JSON (no 'items'). Preview:\n{preview}")

        out = []
        for x in items:
            if isinstance(x, dict) and {"id", "text", "label"}.issubset(x.keys()):
                out.append(
                    {"id": int(x["id"]), "text": str(x["text"]), "label": 1 if int(x["label"]) == 1 else 0}
                )
        if not out:
            raise ValueError(f"[batch {start_id}] parsed JSON but empty payload.")
        return out

    batch = 50
    num_batches = math.ceil(n / batch)
    results: list[dict] = []
    next_id = 1
    for i in range(num_batches):
        want = batch if (i < num_batches - 1) else (n - batch * (num_batches - 1))
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
        time.sleep(0.5)

    if not results:
        raise ValueError("LLM returned no items across all batches")

    rng = random.Random(_seed_from_ds(ds))
    results = _harden_and_shuffle(results, rng=rng, spam_target=0.15, tol=0.05)
    return results

# ---------- Airflow Task ----------
def generate_llm_batches(ds: str | None = None, **context):
    # DagRun conf aus Orchestrator (TriggerDagRunOperator) erlauben
    dag_run = context.get("dag_run")
    conf = (dag_run.conf if dag_run else {}) or {}

    if ds is None:
        ds = conf.get("ds")
    ds = ds or datetime.utcnow().strftime("%Y-%m-%d")

    INCOMING.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    incoming_path = INCOMING / f"{ds}.csv"
    labels_path   = LABELS   / f"{ds}.csv"

    if incoming_path.exists() and labels_path.exists():
        print(f"{incoming_path} and {labels_path} exist; skip.")
        return

    rng = random.Random(_seed_from_ds(ds))
    n = rng.randint(120, 180)  # Variation pro Tag

    rows = _llm_generate(n=n, ds=ds, spam_pct=15)
    df = pd.DataFrame(rows)

    _atomic_write(df[["id", "text"]], incoming_path)
    _atomic_write(df[["id", "label"]], labels_path)
    print(f"Wrote {len(df)} rows -> {incoming_path} & {labels_path}")

# --------------------------------------------------------------------
# Airflow DAG
# --------------------------------------------------------------------
with DAG(
    dag_id="spam_gen_llm",
    default_args=DEFAULT_ARGS,
    schedule=None,                 # wird vom Orchestrator getriggert
    catchup=False,
    tags=["spam", "llm", "generate"],
    description="Generate unlabeled messages + separate labels via LLM (hardened + adversarial seeds)",
) as dag:
    gen = PythonOperator(
        task_id="generate_llm_batches",
        python_callable=generate_llm_batches,
    )