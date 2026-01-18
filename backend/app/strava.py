from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import time
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import settings


class StravaClient:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _request_json(self, method: str, url: str, **kwargs: object) -> dict:
        while True:
            resp = self.session.request(method, url, **kwargs)
            if resp.status_code == 429:
                limit_header = resp.headers.get("X-RateLimit-Limit", "")
                usage_header = resp.headers.get("X-RateLimit-Usage", "")
                try:
                    short_limit = int(limit_header.split(",")[0])
                    short_usage = int(usage_header.split(",")[0])
                except (ValueError, IndexError):
                    short_limit = 0
                    short_usage = 0

                sleep_seconds = 60
                if short_limit and short_usage >= short_limit:
                    sleep_seconds = 900
                time.sleep(sleep_seconds)
                continue

            resp.raise_for_status()
            return resp.json()

    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        url = "https://www.strava.com/oauth/token"
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        return self._request_json("POST", url, data=payload, timeout=30)

    def get_recent_activities(
        self,
        access_token: str,
        per_page: int = 50,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        url = "https://www.strava.com/api/v3/athlete/activities"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"per_page": per_page, "page": page}
        return self._request_json("GET", url, headers=headers, params=params, timeout=30)

    def get_activity_streams(
        self,
        activity_id: int,
        access_token: str,
        streams: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if streams is None:
            streams = ["time", "heartrate"]
        url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"keys": ",".join(streams), "key_by_type": "true"}
        return self._request_json("GET", url, headers=headers, params=params, timeout=30)

    def get_activity_detail(self, activity_id: int, access_token: str) -> Dict[str, Any]:
        url = f"https://www.strava.com/api/v3/activities/{activity_id}"
        headers = {"Authorization": f"Bearer {access_token}"}
        return self._request_json("GET", url, headers=headers, timeout=30)


def build_strava_client() -> StravaClient:
    if not settings.strava_client_id or not settings.strava_client_secret:
        raise RuntimeError("Missing STRAVA_CLIENT_ID or STRAVA_CLIENT_SECRET")
    return StravaClient(settings.strava_client_id, settings.strava_client_secret)


def token_payload_to_expiry(payload: Dict[str, Any]) -> Optional[dt.datetime]:
    expires_at = payload.get("expires_at")
    if not expires_at:
        return None
    return dt.datetime.utcfromtimestamp(expires_at)
