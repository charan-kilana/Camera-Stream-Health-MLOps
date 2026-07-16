"""Generate reproducible synthetic camera-stream telemetry."""    # Comment

from pathlib import Path     # To use later instead of hardscoding for files and folders ex: OUTPUT_PATH = Path("data/camera_stream_telemetry.csv")

import numpy as np    # used for generating random numerical, create arrays, perform numeric operations.
import pandas as pd   # used to create dataframe example excel csv.


RANDOM_SEED = 42    # To generate same data on everyrun. 42 is justt convention inspired from soemhwer.
SAMPLE_COUNT = 5_000    # Sample count
OUTPUT_PATH = Path("data/camera_stream_telemetry.csv")     # Path where generated sample data should be stored.


# Create a function to generate data which takes input as sample count and seed where it returns a pandas dataframe.
def generate_camera_data(        
    sample_count: int = SAMPLE_COUNT, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Create correlated telemetry and a probabilistic stream-failure label."""
    # We're just defining a rng i.e random generator which will be used to generate random number and can be called rng.choice, rng.uniform, rng.normal.
    rng = np.random.default_rng(seed)

    # Hidden operating conditions keep related measurements realistic.
    # creates an array of random choice maintaing the see which generated sample count 5000 records where healthy 65%, degraded 25% and 10% to be critical.

    """
    Instead of generating each metric independently, I first simulated the overall operating condition of each camera stream—healthy, degraded, or critical. Then I generated FPS, latency, packet loss, bitrate, and reconnect count based on that condition. This kept the telemetry internally consistent and much closer to real-world behavior.
    """
    condition = rng.choice(
        ["healthy", "degraded", "critical"],
        size=sample_count,
        p=[0.65, 0.25, 0.10],
    )

    # ranges is a dictionary where it stores key value format. Ex: ranges["Healthy"]["fps"] should generate (24.0 to 30.0). Later numpy will generate the data. 
    ranges = {
        "healthy": {
            "fps": (24.0, 30.0),
            "latency_ms": (20.0, 120.0),
            "packet_loss_percent": (0.0, 1.0),
            "reconnect_count": (0, 2),
        },
        "degraded": {
            "fps": (12.0, 24.0),
            "latency_ms": (100.0, 500.0),
            "packet_loss_percent": (1.0, 8.0),
            "reconnect_count": (0, 5),
        },
        "critical": {
            "fps": (1.0, 12.0),
            "latency_ms": (400.0, 1_500.0),
            "packet_loss_percent": (7.0, 30.0),
            "reconnect_count": (3, 13),
        },
    }

    fps = np.empty(sample_count)
    latency_ms = np.empty(sample_count)
    packet_loss = np.empty(sample_count)
    reconnect_count = np.empty(sample_count, dtype=int)

    for state, state_ranges in ranges.items():
        mask = condition == state
        count = int(mask.sum())
        fps[mask] = rng.uniform(*state_ranges["fps"], count)
        latency_ms[mask] = rng.uniform(*state_ranges["latency_ms"], count)
        packet_loss[mask] = rng.uniform(
            *state_ranges["packet_loss_percent"], count
        )
        reconnect_count[mask] = rng.integers(
            *state_ranges["reconnect_count"], count
        )

    # Bitrate depends on FPS and packet loss instead of being independently random.
    quality_factor = rng.uniform(120.0, 190.0, sample_count)
    bitrate_kbps = fps * quality_factor * (1.0 - packet_loss / 100.0)
    bitrate_kbps += rng.normal(0.0, 120.0, sample_count)
    bitrate_kbps = np.clip(bitrate_kbps, 200.0, 8_000.0)

    uptime_hours = np.where(
        condition == "healthy",
        rng.uniform(24.0, 720.0, sample_count),
        np.where(
            condition == "degraded",
            rng.uniform(2.0, 240.0, sample_count),
            rng.uniform(0.1, 48.0, sample_count),
        ),
    )

    # Several warning signals combine into risk. Random sampling allows degraded
    # streams to recover and apparently healthy streams to fail occasionally.
    risk_score = (
        -3.2
        + 0.14 * (20.0 - fps)
        + 0.0035 * (latency_ms - 150.0)
        + 0.22 * packet_loss
        + 0.32 * reconnect_count
        - 0.0015 * uptime_hours
    )
    failure_probability = 1.0 / (1.0 + np.exp(-risk_score))
    stream_failure = rng.binomial(1, failure_probability)

    data = pd.DataFrame(
        {
            "camera_id": [
                f"CAM-{number:04d}" for number in range(1, sample_count + 1)
            ],
            "fps": np.round(fps, 2),
            "latency_ms": np.round(latency_ms, 2),
            "packet_loss_percent": np.round(packet_loss, 2),
            "bitrate_kbps": np.round(bitrate_kbps, 2),
            "reconnect_count": reconnect_count,
            "uptime_hours": np.round(uptime_hours, 2),
            "stream_failure": stream_failure,
        }
    )
    validate_data(data)
    return data


def validate_data(data: pd.DataFrame) -> None:
    """Reject impossible telemetry before writing the dataset."""
    assert data["fps"].between(0, 30).all()
    assert data["latency_ms"].ge(0).all()
    assert data["packet_loss_percent"].between(0, 100).all()
    assert data["bitrate_kbps"].ge(0).all()
    assert data["reconnect_count"].ge(0).all()
    assert data["uptime_hours"].ge(0).all()
    assert data["stream_failure"].isin([0, 1]).all()


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_camera_data()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(dataset):,} samples at {OUTPUT_PATH}")
    print(f"Stream failure rate: {dataset['stream_failure'].mean():.2%}")
