FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn httpx numpy opencv-python-headless \
    onnxruntime Pillow python-multipart paho-mqtt

# Download Obico failure-detection model
RUN mkdir -p /app/model && \
    wget -q -O /app/model/names \
      "https://raw.githubusercontent.com/TheSpaghettiDetective/obico-server/release/ml_api/model/names" && \
    wget -q -O /tmp/onnx.url \
      "https://raw.githubusercontent.com/TheSpaghettiDetective/obico-server/release/ml_api/model/model-weights.onnx.url" && \
    wget -q -O /app/model/model.onnx "$(cat /tmp/onnx.url | tr -d '\r\n')" && \
    rm /tmp/onnx.url

COPY app.py .

EXPOSE 3333
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3333"]