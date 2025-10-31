# fastapi_app/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# (optional) Buildtools, falls Wheels fehlen
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Nur App-Code kopieren – Modelle werden per Volume gemountet
COPY fastapi_app /app/fastapi_app

# Runtime-Dependencies
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" \
    "numpy>=2,<3" "scikit-learn>=1.4,<2" "scipy>=1.11"

EXPOSE 5000
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "5000"]