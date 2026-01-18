from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class UserCreate(BaseModel):
    name: str


class UserOut(BaseModel):
    id: str
    name: str
    created_at: datetime

    class Config:
        orm_mode = True


class TokenCreate(BaseModel):
    refresh_token: str


class SimulationRequest(BaseModel):
    user_id: str
    gpx_path: str
    limit: int = 50


class SegmentOut(BaseModel):
    cum_dist_miles: float
    grade: float
    predicted_pace_min_per_mile: float
    segment_time_seconds: float


class SimulationResponse(BaseModel):
    predicted_time_seconds: float
    predicted_time_readable: str
    segments: List[SegmentOut]