# fastapi_app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from pathlib import Path

# ---- Load pipeline once at startup (TF-IDF + LogisticRegression) ----
MODEL_PATH = Path("models/logreg_spam_pipeline.pkl")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH.resolve()}")

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

BEST_THRESHOLD = 0.450  # keep your chosen threshold

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    probability_spam: float
    threshold: float

app = FastAPI(title="Spam Classifier API", version="1.0.0")

@app.get("/", tags=["health"])
def root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    # IMPORTANT: we rely on the pipeline's own preprocessing (TfidfVectorizer)
    prob_spam = float(model.predict_proba([text])[0][1])
    pred = 1 if prob_spam >= BEST_THRESHOLD else 0
    label = "spam" if pred == 1 else "ham"

    return PredictResponse(
        prediction=label,
        probability_spam=prob_spam,
        threshold=BEST_THRESHOLD,
    )
