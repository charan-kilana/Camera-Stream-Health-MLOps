"""Train and save the camera-stream failure classifier."""

from pathlib import Path
import pickle    # pickle is a built-in Python module used to save Python objects into a file. ex: In this project, the trained Random Forest model is a Python object. In this project, the trained Random Forest model is a Python object.
 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/camera_stream_telemetry.csv")
MODEL_PATH = Path("models/stream_failure_model.pkl")
FEATURES = [
    "fps",
    "latency_ms",
    "packet_loss_percent",
    "bitrate_kbps",
    "reconnect_count",
    "uptime_hours",
]


def train_model() -> RandomForestClassifier:
    """Train, evaluate, and persist a Random Forest classifier."""
    data = pd.read_csv(DATA_PATH)
    features = data[FEATURES]
    target = data["stream_failure"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(classification_report(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_model()
