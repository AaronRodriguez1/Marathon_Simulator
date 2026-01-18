from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import gpxpy
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def build_baseline_model_with_hr(activities: List[Dict[str, Any]]) -> Tuple[LinearRegression, float]:
    records = []
    for act in activities:
        if act.get("type") != "Run":
            continue
        if not act.get("has_heartrate"):
            continue
        avg_heartrate = act.get("avg_heartrate")
        if avg_heartrate is None:
            continue
        dist_meters = act.get("distance", 0)
        moving_time_seconds = act.get("moving_time", 0)
        if dist_meters <= 0 or moving_time_seconds <= 0:
            continue
        dist_miles = dist_meters * 0.000621371
        elev_gain_m = act.get("total_elevation_gain", 0)
        pace_min_per_mile = (moving_time_seconds / 60.0) / dist_miles
        grade = elev_gain_m / (dist_miles * 1609.34)
        records.append(
            {
                "grade": grade,
                "distance_miles": dist_miles,
                "pace_min_per_mile": pace_min_per_mile,
                "avg_heartrate": avg_heartrate,
            }
        )

    df = pd.DataFrame(records).dropna()
    if df.empty:
        raise ValueError("No activities with HR data to train the model.")
    X = df[["grade", "distance_miles", "avg_heartrate"]].values
    y = df["pace_min_per_mile"].values
    model = LinearRegression()
    model.fit(X, y)
    avg_hr = float(df["avg_heartrate"].mean())
    return model, avg_hr


def haversine(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    R = 6371000
    lat1, lon1 = math.radians(pt1[0]), math.radians(pt1[1])
    lat2, lon2 = math.radians(pt2[0]), math.radians(pt2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def parse_gpx(file_path: str) -> List[Tuple[float, float, float]]:
    with open(file_path, "r") as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    pts = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                pts.append((p.latitude, p.longitude, p.elevation))
    if not pts:
        raise ValueError("GPX file has no track points.")
    return pts


def segment_route(track_pts: List[Tuple[float, float, float]], segment_length: float = 1609.34) -> List[Dict[str, float]]:
    segs = []
    cum_dist_meters = 0.0
    seg_points = [track_pts[0]]
    for i in range(1, len(track_pts)):
        prev = track_pts[i - 1]
        curr = track_pts[i]
        d = haversine(prev[:2], curr[:2])
        cum_dist_meters += d
        seg_points.append(curr)
        if cum_dist_meters >= segment_length:
            elev_diff = seg_points[-1][2] - seg_points[0][2]
            avg_grade = elev_diff / cum_dist_meters
            segs.append({"grade": avg_grade, "dist_miles": cum_dist_meters * 0.000621371})
            cum_dist_meters = 0.0
            seg_points = [track_pts[i]]
    if cum_dist_meters > 0:
        elev_diff = seg_points[-1][2] - seg_points[0][2]
        avg_grade = elev_diff / cum_dist_meters
        segs.append({"grade": avg_grade, "dist_miles": cum_dist_meters * 0.000621371})
    return segs


def simulate_route(
    segments: List[Dict[str, float]],
    model: LinearRegression,
    avg_heartrate: float,
) -> pd.DataFrame:
    records = []
    cum_dist_miles = 0.0
    total_time_seconds = 0.0
    for seg in segments:
        dist_miles = seg["dist_miles"]
        grade = seg["grade"]
        cum_dist_miles += dist_miles
        X = np.array([[grade, cum_dist_miles, avg_heartrate]])
        predicted_pace_min_per_mile = model.predict(X)[0]
        predicted_time_seconds = predicted_pace_min_per_mile * 60 * dist_miles
        total_time_seconds += predicted_time_seconds
        records.append(
            {
                "cum_dist_miles": cum_dist_miles,
                "grade": grade,
                "predicted_pace_min_per_mile": float(predicted_pace_min_per_mile),
                "segment_time_seconds": float(predicted_time_seconds),
            }
        )
    df = pd.DataFrame(records)
    df["total_predicted_time_seconds"] = total_time_seconds
    return df