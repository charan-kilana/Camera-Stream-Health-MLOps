"""FastAPI service for camera-stream failure prediction."""

from pathlib import Path
import pickle

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/stream_failure_model.pkl")
app = FastAPI(title="Camera Stream Health API", version="1.0.0")


class CameraTelemetry(BaseModel):
    fps: float = Field(ge=0, le=30)
    latency_ms: float = Field(ge=0)
    packet_loss_percent: float = Field(ge=0, le=100)
    bitrate_kbps: float = Field(ge=0)
    reconnect_count: int = Field(ge=0)
    uptime_hours: float = Field(ge=0)


def load_model():
    """Load the trained model when it is available."""
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


model = load_model()


@app.get("/health")
def health():
    return {"status": "healthy" if model is not None else "model_unavailable"}


@app.post("/predict")
def predict(data: CameraTelemetry):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Run train.py first.")

    features = np.array(
        [[
            data.fps,
            data.latency_ms,
            data.packet_loss_percent,
            data.bitrate_kbps,
            data.reconnect_count,
            data.uptime_hours,
        ]]
    )
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    risk_level = (
        "high" if probability >= 0.7 else "medium" if probability >= 0.3 else "low"
    )
    return {
        "stream_failure": prediction,
        "failure_probability": round(probability, 4),
        "risk_level": risk_level,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
