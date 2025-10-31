# fastapi_app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json, pickle

# ----- Model discovery via pointer-file -----
MODEL_DIR = Path("/app/models")
POINTER = MODEL_DIR / "production_model.txt"   # enthält z.B. "model_current.pkl"
THRESHOLD_FILE = MODEL_DIR / "threshold.json"  # {"threshold": 0.45}

def load_model_and_threshold():
    # 1) Modellpfad bestimmen
    if POINTER.exists():
        name = POINTER.read_text().strip()
        model_path = MODEL_DIR / name
    else:
        # Fallback: erstes .pkl im Ordner
        pkl_files = list(MODEL_DIR.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No model .pkl found in {MODEL_DIR}")
        model_path = pkl_files[0]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # 2) Modell laden
    with open(model_path, "rb") as f:
        mdl = pickle.load(f)

    # 3) Threshold laden (Fallback 0.5)
    thr = 0.5
    if THRESHOLD_FILE.exists():
        try:
            thr = float(json.loads(THRESHOLD_FILE.read_text()).get("threshold", thr))
        except Exception:
            pass

    return mdl, thr, model_path.name

model, BEST_THRESHOLD, model_name = load_model_and_threshold()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    probability_spam: float
    threshold: float
    model: str

app = FastAPI(title="Spam Classifier API", version="1.0.0")

# CORS: Deine GitHub-Page darf anfragen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://thiev980.github.io"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "model": model_name, "threshold": BEST_THRESHOLD}

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    prob_spam = float(model.predict_proba([text])[0][1])
    pred = 1 if prob_spam >= BEST_THRESHOLD else 0
    label = "spam" if pred == 1 else "ham"
    return PredictResponse(
        prediction=label,
        probability_spam=prob_spam,
        threshold=BEST_THRESHOLD,
        model=model_name,
    )