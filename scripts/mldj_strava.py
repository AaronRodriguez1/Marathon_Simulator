import requests
import os
from dotenv import load_dotenv


# Load API keys from .env
load_dotenv()
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")

# Debugging: Print values (Remove these after testing)
print(f"Client ID: {CLIENT_ID}")
print(f"Client Secret: {CLIENT_SECRET}")
print(f"Refresh Token: {REFRESH_TOKEN}")

# Function to get a new access token
def get_access_token():
    url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token"
    }
    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        token_data = response.json()
        return token_data["access_token"]
    else:
        print("Error getting access token:", response.json())
        return None

# Function to get recent activities
def get_recent_activities():
    access_token = get_access_token()
    if not access_token:
        return None

    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()  # List of activities
    else:
        print("Error fetching activities:", response.json())
        return None

# Run the script
if __name__ == "__main__":
    activities = get_recent_activities()
    if activities:
        for i, act in enumerate(activities[:5]):  # Print first 5 activities
            print(f"{i+1}. {act['name']} - {act['distance']}m in {act['moving_time']}s")

