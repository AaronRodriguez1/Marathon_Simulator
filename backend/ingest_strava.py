import datetime as dt
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv, find_dotenv
from sqlalchemy.orm import Session

from app.db import Base, SessionLocal, engine
from app.features import compute_stream_features
from app.models import Activity, ActivityFeature, ActivityStream
from app.strava import build_strava_client


def parse_strava_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def extract_latlng(pair: Optional[List[float]]) -> tuple[Optional[float], Optional[float]]:
    if not pair or len(pair) != 2:
        return None, None
    return pair[0], pair[1]


def upsert_activity(db: Session, activity: Dict[str, Any]) -> Activity:
    activity_id = activity.get("id")
    row = db.query(Activity).filter(Activity.id == activity_id).first()
    start_lat, start_lng = extract_latlng(activity.get("start_latlng"))
    end_lat, end_lng = extract_latlng(activity.get("end_latlng"))

    payload = {
        "id": activity_id,
        "name": activity.get("name"),
        "type": activity.get("type"),
        "start_date": parse_strava_datetime(activity.get("start_date")),
        "timezone": activity.get("timezone"),
        "distance": activity.get("distance"),
        "moving_time": activity.get("moving_time"),
        "elapsed_time": activity.get("elapsed_time"),
        "total_elevation_gain": activity.get("total_elevation_gain"),
        "average_speed": activity.get("average_speed"),
        "max_speed": activity.get("max_speed"),
        "average_heartrate": activity.get("average_heartrate"),
        "max_heartrate": activity.get("max_heartrate"),
        "average_cadence": activity.get("average_cadence"),
        "calories": activity.get("calories"),
        "has_heartrate": activity.get("has_heartrate"),
        "start_lat": start_lat,
        "start_lng": start_lng,
        "end_lat": end_lat,
        "end_lng": end_lng,
        "external_id": activity.get("external_id"),
        "raw": activity,
    }

    if row:
        for key, value in payload.items():
            setattr(row, key, value)
    else:
        row = Activity(**payload)
        db.add(row)

    return row


def upsert_streams(db: Session, activity_id: int, streams: Dict[str, Any]) -> None:
    for stream_type, stream_payload in streams.items():
        if not isinstance(stream_payload, dict):
            continue
        row = (
            db.query(ActivityStream)
            .filter(ActivityStream.activity_id == activity_id, ActivityStream.stream_type == stream_type)
            .first()
        )
        payload = {
            "activity_id": activity_id,
            "stream_type": stream_type,
            "data": stream_payload.get("data"),
            "original_size": stream_payload.get("original_size"),
            "resolution": stream_payload.get("resolution"),
        }
        if row:
            for key, value in payload.items():
                setattr(row, key, value)
        else:
            db.add(ActivityStream(**payload))


def upsert_features(db: Session, activity_id: int, features: Dict[str, float]) -> None:
    row = db.query(ActivityFeature).filter(ActivityFeature.activity_id == activity_id).first()
    payload = {
        "activity_id": activity_id,
        "features": features,
    }
    if row:
        for key, value in payload.items():
            setattr(row, key, value)
    else:
        db.add(ActivityFeature(**payload))


def fetch_all_activities(access_token: str, per_page: int = 200, max_pages: int = 50) -> List[Dict[str, Any]]:
    client = build_strava_client()
    activities: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        batch = client.get_recent_activities(access_token, per_page=per_page, page=page)
        if not batch:
            break
        activities.extend(batch)
        if len(batch) < per_page:
            break
    return activities


def main() -> None:
    load_dotenv(find_dotenv(filename=".env", usecwd=True))
    refresh_token = os.getenv("STRAVA_REFRESH_TOKEN")
    if not refresh_token:
        raise RuntimeError("Missing STRAVA_REFRESH_TOKEN in environment")
    skip_existing = os.getenv("SKIP_EXISTING", "true").lower() != "false"
    commit_every = int(os.getenv("COMMIT_EVERY", "5"))

    client = build_strava_client()
    token_payload = client.refresh_access_token(refresh_token)
    access_token = token_payload.get("access_token")
    if not access_token:
        raise RuntimeError("Failed to get access token")

    Base.metadata.create_all(bind=engine)
    activities = fetch_all_activities(access_token)

    with SessionLocal() as db:
        total = len(activities)
        for idx, act in enumerate(activities, start=1):
            if idx == 1 or idx % 10 == 0:
                print(f"Processing activity {idx}/{total} (id={act.get('id')})")
            if skip_existing:
                existing = db.query(Activity).filter(Activity.id == act["id"]).first()
                if existing:
                    continue

            detail = client.get_activity_detail(act["id"], access_token)
            upsert_activity(db, detail)
            if detail.get("type") == "Run":
                streams = client.get_activity_streams(
                    detail["id"],
                    access_token,
                    streams=[
                        "time",
                        "distance",
                        "heartrate",
                        "cadence",
                        "velocity_smooth",
                        "altitude",
                        "latlng",
                        "grade_smooth",
                    ],
                )
                upsert_streams(db, detail["id"], streams)
                features = compute_stream_features(streams)
                if features:
                    upsert_features(db, detail["id"], features)
            time.sleep(0.2)
            if idx % commit_every == 0:
                db.commit()
        db.commit()

    print(f"Stored {len(activities)} activities in the database.")


if __name__ == "__main__":
    main()
