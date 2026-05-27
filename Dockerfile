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

# Embedding-Modell zentral als ENV → Pre-Cache UND Runtime (src/embeddings.py) nutzen
# denselben Namen, keine Drift. Multilingual (DE/EN), 384-dim.
ENV EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
# Pre-cache embedding model (optional — speeds up first request)
RUN python -c "import os;\ntry:\n from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL']); print('Model cached successfully')\nexcept Exception as e:\n print(f'Model cache skipped: {e}')" || true

COPY src/ src/
COPY docs/ docs/
COPY app.py .

EXPOSE 8000
EXPOSE 7864

ENV GRADIO_SERVER_NAME=0.0.0.0

# Default: FastAPI REST API (ui-Service überschreibt mit: python app.py)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
