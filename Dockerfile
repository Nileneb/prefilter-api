FROM python:3.12-slim

WORKDIR /app

# Dependencies first (layer caching)
# requirements.lock = pip freeze output → reproducible builds
# Fallback: requirements.txt (loose pins) when lock doesn't exist yet
COPY requirements.txt .
COPY requirements.loc[k] .
# Install CPU-only torch first — prevents pip from pulling CUDA packages (server has no GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN if [ -f requirements.lock ]; then \
      pip install --no-cache-dir -r requirements.lock; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi

# Pre-cache embedding model (optional — speeds up first request)
RUN python -c "try:\n from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Model cached successfully')\nexcept Exception as e:\n print(f'Model cache skipped: {e}')" || true

COPY src/ src/
COPY docs/ docs/
COPY app.py .

EXPOSE 8000
EXPOSE 7864

ENV GRADIO_SERVER_NAME=0.0.0.0

# Default: FastAPI REST API (ui-Service überschreibt mit: python app.py)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
