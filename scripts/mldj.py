import requests
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
import math

# Load Strava API credentials
load_dotenv()
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")

# 1. STRAVA DATA FETCHING

def get_access_token():
    """
    Exchange refresh token for a new access token.
    """
    url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token"
    }
    resp = requests.post(url, data=payload)
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_recent_activities(n=50):
    """
    Retrieve up to n recent activities from Strava.
    """
    token = get_access_token()
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"per_page": n}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()

def get_activity_streams(activity_id, access_token, streams=["time", "heartrate"]):
    """
    Retrieves the specified streams for a given activity.
    """
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"keys": ",".join(streams), "key_by_type": "true"}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()

def get_recent_runs_with_hr(n=50):
    """
    Retrieves up to n recent 'Run' activities with heart rate data.
    """
    token = get_access_token()
    all_activities = get_recent_activities(n)
    run_activities_with_hr = []
    for act in all_activities:
        if act.get("type") == "Run" and act.get("has_heartrate"):
            streams = get_activity_streams(act["id"], token)
            if "heartrate" in streams and "time" in streams and act.get("distance", 0) > 0 and act.get("moving_time", 0) > 0:
                time_data = streams["time"]["data"]
                heartrate_data = streams["heartrate"]["data"]
                avg_hr = np.mean(heartrate_data) if heartrate_data else None
                act["avg_heartrate"] = avg_hr
                run_activities_with_hr.append(act)
            else:
                act["avg_heartrate"] = None
                run_activities_with_hr.append(act)
        elif act.get("type") == "Run":
            act["avg_heartrate"] = None
            run_activities_with_hr.append(act)
    return run_activities_with_hr

# 2. BASELINE MODEL TRAINING (Pace in minutes/mile with heart rate)

def build_baseline_model_with_hr(activities):
    """
    Fit a linear regression model: pace (min/mile) ~ grade + distance_miles + avg_heartrate,
    using only 'Run' activities with heart rate data.
    """
    records = []
    for act in activities:
        if act.get("type") == "Run" and act.get("has_heartrate") and act.get("avg_heartrate") is not None:
            dist_meters = act.get("distance", 0)
            if dist_meters <= 0:
                continue
            dist_miles = dist_meters * 0.000621371
            elev_gain_m = act.get("total_elevation_gain", 0)
            moving_time_seconds = act.get("moving_time", 0)
            pace_min_per_mile = (moving_time_seconds / 60.0) / dist_miles if dist_miles > 0 else np.nan
            grade = elev_gain_m / (dist_miles * 1609.34) if dist_miles > 0 else 0
            avg_heartrate = act.get("avg_heartrate")

            if not np.isnan(pace_min_per_mile) and avg_heartrate is not None:
                records.append({
                    "grade": grade,
                    "distance_miles": dist_miles,
                    "pace_min_per_mile": pace_min_per_mile,
                    "avg_heartrate": avg_heartrate
                })
    df = pd.DataFrame(records).dropna()
    X = df[["grade", "distance_miles", "avg_heartrate"]].values
    y = df["pace_min_per_mile"].values
    model = LinearRegression()
    model.fit(X, y)
    return model

# 3. ROUTE PARSING & SEGMENTATION

def haversine(pt1, pt2):
    """
    Calculate haversine distance (meters) between two (lat, lon) points.
    """
    R = 6371000  # Earth radius in meters
    lat1, lon1 = math.radians(pt1[0]), math.radians(pt1[1])
    lat2, lon2 = math.radians(pt2[0]), math.radians(pt2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def parse_gpx(file_path):
    """
    Parse a GPX file and return a list of (lat, lon, elevation) track points.
    """
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    pts = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                pts.append((p.latitude, p.longitude, p.elevation))
    return pts


def segment_route(track_pts, segment_length=1609.34): # Segment length in miles
    """
    Break the route into fixed-length segments (meters - will be converted to miles).
    Returns list of dicts with avg grade and segment distance (in miles).
    """
    segs = []
    cum_dist_meters = 0.0
    seg_points = [track_pts[0]]
    for i in range(1, len(track_pts)):
        prev = track_pts[i-1]
        curr = track_pts[i]
        d = haversine(prev[:2], curr[:2])
        cum_dist_meters += d
        seg_points.append(curr)
        if cum_dist_meters >= segment_length:
            # compute avg grade for segment
            elev_diff = seg_points[-1][2] - seg_points[0][2]
            avg_grade = elev_diff / cum_dist_meters
            segs.append({"grade": avg_grade, "dist_miles": cum_dist_meters * 0.000621371})
            # reset for next segment
            cum_dist_meters = 0.0
            seg_points = [track_pts[i]]
    # Handle any remaining segment
    if cum_dist_meters > 0:
        elev_diff = seg_points[-1][2] - seg_points[0][2]
        avg_grade = elev_diff / cum_dist_meters
        segs.append({"grade": avg_grade, "dist_miles": cum_dist_meters * 0.000621371})
    return segs

# 4. SIMULATION & VISUALIZATION (Pace in minutes/mile)

def simulate_route(segments, model):
    """
    Use the trained model to predict pace and time on each segment (in min/mile).
    Returns DataFrame with cumulative distance (miles), grade, predicted pace (min/mile),
    and predicted time (seconds) for each segment.
    """
    records = []
    cum_dist_miles = 0.0
    total_time_seconds = 0.0
    for seg in segments:
        dist_miles = seg["dist_miles"]
        grade = seg["grade"]
        cum_dist_miles += dist_miles
        # Note: We are not predicting heart rate here, just pace based on grade, distance, and historical HR influence.
        X = np.array([[grade, cum_dist_miles, model.coef_[2]]]) # Assuming HR coefficient is the third one
        predicted_pace_min_per_mile = model.predict(X)[0]
        predicted_time_seconds = predicted_pace_min_per_mile * 60 * dist_miles
        total_time_seconds += predicted_time_seconds
        records.append({
            "cum_dist_miles": cum_dist_miles,
            "grade": grade,
            "predicted_pace_min_per_mile": predicted_pace_min_per_mile,
            "segment_time_seconds": predicted_time_seconds
        })
    df = pd.DataFrame(records)
    df["total_predicted_time_seconds"] = total_time_seconds
    return df


def plot_simulation(df):
    """
    Plot predicted pace (min/mile) over distance (miles) and overlay elevation/grade.
    """
    fig, ax1 = plt.subplots()
    ax1.plot(df["cum_dist_miles"], df["predicted_pace_min_per_mile"], label="Predicted Pace (min/mile)")
    ax1.set_xlabel("Distance (miles)")
    ax1.set_ylabel("Pace (min/mile)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(df["cum_dist_miles"], df["grade"], label="Avg Grade", linestyle='--')
    ax2.set_ylabel("Grade")
    ax2.legend(loc="upper right")
    plt.title("Virtual Marathon Performance Simulation")
    plt.tight_layout()
    plt.show()

# 5. MAIN PIPELINE

def main():
    run_activities = get_recent_runs_with_hr(n=50)

    print("Recent Runs, Paces (minutes/mile), and Average Heart Rate:")
    for activity in run_activities:
        activity_type = activity.get("type")
        distance_meters = activity.get("distance")
        moving_time_seconds = activity.get("moving_time")
        avg_hr = activity.get("avg_heartrate")

        pace_str = "N/A (No distance or time data)"
        if distance_meters and moving_time_seconds and distance_meters > 0:
            distance_miles = distance_meters * 0.000621371
            pace_minutes_per_mile = (moving_time_seconds / 60.0) / distance_miles
            pace_str = f"{pace_minutes_per_mile:.2f} min/mile"

        hr_str = f"{avg_hr:.0f} bpm" if avg_hr is not None else "N/A (No HR data)"
        print(f"Activity: {activity_type}, Pace: {pace_str}, Avg HR: {hr_str}")

    print("\n--- Proceeding with model training and route simulation (using only run data with heart rate) ---")

    model = build_baseline_model_with_hr(run_activities)

    gpx_path = "boston-marathon-course.gpx"  # e.g., Boston Marathon GPX
    track_points = parse_gpx(gpx_path)
    segments = segment_route(track_points, segment_length=1609.34)

    sim_df = simulate_route(segments, model)
    plot_simulation(sim_df)

    total_predicted_time_seconds = sim_df["total_predicted_time_seconds"].iloc[-1]
    total_predicted_time = pd.Timedelta(seconds=total_predicted_time_seconds)

    print(f"\nPredicted Marathon Completion Time: {total_predicted_time}")

if __name__ == "__main__":
    main()