from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    strava_token = relationship("StravaToken", back_populates="user", uselist=False)


class StravaToken(Base):
    __tablename__ = "strava_tokens"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    refresh_token = Column(String, nullable=False)
    access_token = Column(String, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    user = relationship("User", back_populates="strava_token")


class Activity(Base):
    __tablename__ = "activities"

    id = Column(BigInteger, primary_key=True)
    name = Column(String, nullable=True)
    type = Column(String, nullable=True)
    start_date = Column(DateTime, nullable=True)
    timezone = Column(String, nullable=True)
    distance = Column(Float, nullable=True)
    moving_time = Column(Integer, nullable=True)
    elapsed_time = Column(Integer, nullable=True)
    total_elevation_gain = Column(Float, nullable=True)
    average_speed = Column(Float, nullable=True)
    max_speed = Column(Float, nullable=True)
    average_heartrate = Column(Float, nullable=True)
    max_heartrate = Column(Float, nullable=True)
    average_cadence = Column(Float, nullable=True)
    calories = Column(Float, nullable=True)
    has_heartrate = Column(Boolean, nullable=True)
    start_lat = Column(Float, nullable=True)
    start_lng = Column(Float, nullable=True)
    end_lat = Column(Float, nullable=True)
    end_lng = Column(Float, nullable=True)
    external_id = Column(String, nullable=True)
    raw = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    streams = relationship("ActivityStream", back_populates="activity")


class ActivityStream(Base):
    __tablename__ = "activity_streams"

    id = Column(Integer, primary_key=True)
    activity_id = Column(BigInteger, ForeignKey("activities.id"), nullable=False)
    stream_type = Column(String, nullable=False)
    data = Column(JSON, nullable=True)
    original_size = Column(Integer, nullable=True)
    resolution = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    activity = relationship("Activity", back_populates="streams")


class ActivityFeature(Base):
    __tablename__ = "activity_features"

    id = Column(Integer, primary_key=True)
    activity_id = Column(BigInteger, ForeignKey("activities.id"), nullable=False, unique=True)
    features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
