# fastapi_app/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# (optional) System-Buildtools, falls Wheels fehlen
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Code + Modell kopieren (Variante A: Modell ins Image)
COPY fastapi_app /app/fastapi_app
COPY models /app/models

# Runtime-Dependencies: NumPy 2.x + kompatibles sklearn
RUN pip install --no-cache-dir \
    "fastapi" "uvicorn[standard]" \
    "numpy>=2,<3" "scikit-learn>=1.4,<2" "scipy>=1.11"

EXPOSE 5000
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "5000"]