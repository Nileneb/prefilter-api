FROM python:3.12-slim

WORKDIR /app

# Dependencies first (layer caching)
# requirements.lock = pip freeze output → reproducible builds
# Fallback: requirements.txt (loose pins) when lock doesn't exist yet
COPY requirements.txt .
COPY requirements.loc[k] .
RUN if [ -f requirements.lock ]; then \
      pip install --no-cache-dir -r requirements.lock; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi

COPY src/ src/
COPY app.py .

EXPOSE 8000
EXPOSE 7864

ENV GRADIO_SERVER_NAME=0.0.0.0

# Default: FastAPI REST API (ui-Service überschreibt mit: python app.py)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
