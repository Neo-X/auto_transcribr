#!/usr/bin/env python3
import os
import sys
import requests
from datetime import datetime, timedelta
from pathlib import Path

ACCOUNT_ID = os.environ.get("ZOOM_ACCOUNT_ID")
CLIENT_ID = os.environ.get("ZOOM_CLIENT_ID")
CLIENT_SECRET = os.environ.get("ZOOM_CLIENT_SECRET")
DOWNLOAD_DIR = Path.home() / "Downloads" / "zoom_recordings"


def get_token():
    resp = requests.post(
        "https://zoom.us/oauth/token",
        params={"grant_type": "account_credentials", "account_id": ACCOUNT_ID},
        auth=(CLIENT_ID, CLIENT_SECRET),
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def list_recordings(token):
    today = datetime.utcnow().date()
    from_date = today - timedelta(days=14)
    headers = {"Authorization": f"Bearer {token}"}
    meetings = []
    next_page = None

    while True:
        params = {"from": from_date.isoformat(), "to": today.isoformat(), "page_size": 300}
        if next_page:
            params["next_page_token"] = next_page
        resp = requests.get("https://api.zoom.us/v2/users/me/recordings", headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        meetings.extend(data.get("meetings", []))
        next_page = data.get("next_page_token")
        if not next_page:
            break

    return meetings


def download_file(url, dest_path, token):
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def main():
    if not all([ACCOUNT_ID, CLIENT_ID, CLIENT_SECRET]):
        print("Set ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET env vars.")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("Authenticating...")
    token = get_token()

    print("Fetching recordings from the last 2 weeks...")
    meetings = list_recordings(token)
    print(f"Found {len(meetings)} meetings.")

    downloaded, skipped = 0, 0
    for meeting in meetings:
        topic = meeting.get("topic", "untitled").replace("/", "-").replace(":", "-")
        start = meeting.get("start_time", "")[:10]

        for rec in meeting.get("recording_files", []):
            if rec.get("file_type") != "MP4" or rec.get("status") != "completed":
                continue

            filename = f"{start}_{topic}_{rec['id']}.mp4"
            dest = DOWNLOAD_DIR / filename

            if dest.exists():
                print(f"  [skip] {filename}")
                skipped += 1
                continue

            url = rec.get("download_url")
            print(f"  [download] {filename}")
            download_file(url, dest, token)
            downloaded += 1

    print(f"\nDone. Downloaded: {downloaded}, Skipped (already exist): {skipped}")
    print(f"Files saved to: {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()
