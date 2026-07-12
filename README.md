# Camera Stream Health MLOps

A basic end-to-end machine learning project that predicts whether a surveillance
camera stream is at risk of failure from operational telemetry.

The repository follows a simple MLOps workflow:

1. Generate a reproducible synthetic dataset.
2. Train and evaluate a Random Forest classifier.
3. Save the trained model.
4. Serve predictions through FastAPI.

## Model inputs

- Frames per second (`fps`)
- Network latency in milliseconds
- Packet loss percentage
- Video bitrate in Kbps
- Recent reconnection count
- Current stream uptime in hours

The target, `stream_failure`, indicates whether a stream fails within the chosen
prediction window. The dataset is a domain-informed synthetic approximation for
demonstrating the ML workflow, not production camera telemetry.

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
python generate_data.py
python train.py
python api.py
```

Open `http://localhost:8000/docs` to test the API.

## Run with Docker

Generate the dataset and train the model before building the image:

```bash
python generate_data.py
python train.py
docker build -t camera-stream-health-api:latest .
docker run --rm -p 8000:8000 --name camera-health-api camera-stream-health-api:latest
```

Open `http://localhost:8000/docs` for Swagger UI or check container health:

```bash
curl http://localhost:8000/health
docker ps
```

The Docker image includes `models/stream_failure_model.pkl`. The model is kept
out of Git and must be trained or downloaded before each image build.

Example request:

```json
{
  "fps": 8.0,
  "latency_ms": 650.0,
  "packet_loss_percent": 12.0,
  "bitrate_kbps": 700.0,
  "reconnect_count": 5,
  "uptime_hours": 3.0
}
```

Example response:

```json
{
  "stream_failure": 1,
  "failure_probability": 0.89,
  "risk_level": "high"
}
```
