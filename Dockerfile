FROM python:3.12-slim

WORKDIR /app

# Dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY modules/ modules/
COPY app.py .

EXPOSE 8000
EXPOSE 7864

ENV GRADIO_SERVER_NAME=0.0.0.0

# Default: FastAPI REST API (ui-Service überschreibt mit: python app.py)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
