import os
from dotenv import load_dotenv
import requests

load_dotenv()
client_id = os.getenv('STRAVA_CLIENT_ID')
client_secret = os.getenv('STRAVA_CLIENT_SECRET')
refresh_token = os.getenv('STRAVA_REFRESH_TOKEN')

if not all([client_id, client_secret, refresh_token]):
    raise SystemExit('Missing STRAVA_CLIENT_ID/SECRET/REFRESH_TOKEN in .env')

resp = requests.post(
    'https://www.strava.com/oauth/token',
    data={
        'client_id': client_id,
        'client_secret': client_secret,
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token',
    },
    timeout=30,
)
resp.raise_for_status()
access_token = resp.json()['access_token']

activities = requests.get(
    'https://www.strava.com/api/v3/athlete/activities',
    headers={'Authorization': f'Bearer {access_token}'},
    params={'per_page': 1, 'page': 1},
    timeout=30,
)
activities.raise_for_status()
activity = None
for act in activities.json():
    if act.get('type') == 'Run' and act.get('has_heartrate'):
        activity = act
        break

if activity is None:
    raise SystemExit('No Run activities with heart rate found in this page.')

activity_id = activity['id']
detail = requests.get(
    f'https://www.strava.com/api/v3/activities/{activity_id}',
    headers={'Authorization': f'Bearer {access_token}'},
    timeout=30,
)
detail.raise_for_status()

detail_json = detail.json()

streams_requested = [
    "time",
    "distance",
    "latlng",
    "altitude",
    "velocity_smooth",
    "heartrate",
    "cadence",
    "grade_smooth",
]
streams = requests.get(
    f'https://www.strava.com/api/v3/activities/{activity_id}/streams',
    headers={'Authorization': f'Bearer {access_token}'},
    params={'keys': ','.join(streams_requested), 'key_by_type': 'true'},
    timeout=30,
)
streams.raise_for_status()
streams_json = streams.json()

summary_keys = set(activity.keys())
detail_keys = set(detail_json.keys())
detail_only = sorted(detail_keys - summary_keys)

print('Summary keys:')
print(sorted(summary_keys))
print('\nDetail-only keys:')
print(detail_only)
print('\nStream keys returned:')
print(sorted(streams_json.keys()))
print('\nStream keys missing (requested but not returned):')
print(sorted([k for k in streams_requested if k not in streams_json]))
