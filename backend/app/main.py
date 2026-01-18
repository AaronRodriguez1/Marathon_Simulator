from __future__ import annotations

import datetime as dt
import os
from typing import Generator

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import models, pipeline, schemas
from .config import settings
from .db import Base, SessionLocal, engine
from .strava import build_strava_client, token_payload_to_expiry


Base.metadata.create_all(bind=engine)

app = FastAPI(title="MLDJ API", version="0.1.0")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/users", response_model=schemas.UserOut)
def create_user(payload: schemas.UserCreate, db: Session = Depends(get_db)) -> models.User:
    user = models.User(name=payload.name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/users/{user_id}/tokens")
def upsert_tokens(
    user_id: str,
    payload: schemas.TokenCreate,
    db: Session = Depends(get_db),
) -> dict:
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    token = db.query(models.StravaToken).filter(models.StravaToken.user_id == user_id).first()
    if not token:
        token = models.StravaToken(user_id=user_id, refresh_token=payload.refresh_token)
        db.add(token)
    else:
        token.refresh_token = payload.refresh_token
        token.access_token = None
        token.expires_at = None
    db.commit()
    return {"status": "ok"}


@app.post("/simulate", response_model=schemas.SimulationResponse)
def simulate(
    payload: schemas.SimulationRequest,
    db: Session = Depends(get_db),
) -> schemas.SimulationResponse:
    if not os.path.exists(payload.gpx_path):
        raise HTTPException(status_code=400, detail="GPX path not found")

    token = db.query(models.StravaToken).filter(models.StravaToken.user_id == payload.user_id).first()
    if not token:
        raise HTTPException(status_code=404, detail="Strava token not found for user")

    client = build_strava_client()
    token_payload = client.refresh_access_token(token.refresh_token)
    token.access_token = token_payload.get("access_token")
    token.expires_at = token_payload_to_expiry(token_payload)
    token.updated_at = dt.datetime.utcnow()
    db.commit()

    activities = client.get_recent_activities(token.access_token, per_page=payload.limit)
    run_activities = []
    for act in activities:
        if act.get("type") != "Run":
            continue
        if not act.get("has_heartrate"):
            act["avg_heartrate"] = None
            run_activities.append(act)
            continue
        streams = client.get_activity_streams(act["id"], token.access_token)
        if "heartrate" in streams and "time" in streams and act.get("distance", 0) > 0:
            hr_data = streams["heartrate"]["data"]
            act["avg_heartrate"] = float(sum(hr_data) / len(hr_data)) if hr_data else None
        else:
            act["avg_heartrate"] = None
        run_activities.append(act)

    model, avg_hr = pipeline.build_baseline_model_with_hr(run_activities)
    track_points = pipeline.parse_gpx(payload.gpx_path)
    segments = pipeline.segment_route(track_points)
    sim_df = pipeline.simulate_route(segments, model, avg_hr)

    total_predicted_time_seconds = float(sim_df["total_predicted_time_seconds"].iloc[-1])
    predicted_readable = str(dt.timedelta(seconds=total_predicted_time_seconds))

    return schemas.SimulationResponse(
        predicted_time_seconds=total_predicted_time_seconds,
        predicted_time_readable=predicted_readable,
        segments=[schemas.SegmentOut(**row) for row in sim_df.drop(columns=["total_predicted_time_seconds"]).to_dict("records")],
    )
