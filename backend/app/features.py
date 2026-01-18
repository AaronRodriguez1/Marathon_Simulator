from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _safe_array(streams: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    payload = streams.get(key)
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not data:
        return None
    return np.array(data, dtype=float)


def _pace_from_velocity(velocity: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        pace_sec_per_mile = 1609.34 / velocity
    pace_sec_per_mile = pace_sec_per_mile[~np.isinf(pace_sec_per_mile)]
    return pace_sec_per_mile


def compute_stream_features(streams: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}

    hr = _safe_array(streams, "heartrate")
    if hr is not None and hr.size > 0:
        features["hr_mean"] = float(np.mean(hr))
        features["hr_std"] = float(np.std(hr))
        features["hr_max"] = float(np.max(hr))

    cadence = _safe_array(streams, "cadence")
    if cadence is not None and cadence.size > 0:
        features["cadence_mean"] = float(np.mean(cadence))
        features["cadence_std"] = float(np.std(cadence))

    grade = _safe_array(streams, "grade_smooth")
    if grade is not None and grade.size > 0:
        features["grade_mean"] = float(np.mean(grade))
        features["grade_std"] = float(np.std(grade))

    altitude = _safe_array(streams, "altitude")
    if altitude is not None and altitude.size > 1:
        diffs = np.diff(altitude)
        features["elevation_gain_stream"] = float(np.sum(diffs[diffs > 0]))

    velocity = _safe_array(streams, "velocity_smooth")
    if velocity is not None and velocity.size > 0:
        pace_sec_per_mile = _pace_from_velocity(velocity)
        if pace_sec_per_mile.size > 0:
            features["pace_mean_min_per_mile"] = float(np.mean(pace_sec_per_mile) / 60.0)
            features["pace_std_min_per_mile"] = float(np.std(pace_sec_per_mile) / 60.0)

            mid = pace_sec_per_mile.size // 2
            if mid > 0:
                first_half = pace_sec_per_mile[:mid]
                second_half = pace_sec_per_mile[mid:]
                if first_half.size > 0 and second_half.size > 0:
                    features["pace_drift_min_per_mile"] = float(
                        (np.mean(second_half) - np.mean(first_half)) / 60.0
                    )

    return features