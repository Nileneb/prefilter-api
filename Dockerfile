FROM python:3.12-slim

WORKDIR /app

# Dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 7864

ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "main.py"]
