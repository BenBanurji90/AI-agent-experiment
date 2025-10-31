"""Spotify tool-calling example exposing search and audio feature lookups.

Available tools:
- `spotify_search_tracks` for free-text track search with links and previews.
- `spotify_track_features` for tempo/danceability/energy metrics.

"""

import argparse
import asyncio
import base64
import os
import time
from datetime import date
from typing import Any, Dict, List, Optional

import requests
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

load_dotenv()

## Things I learnt:
### 1) Codex does not have current access to the latest API documentation (in this case Spotify
### 2) The agent is still limited by knowledge cuuoffs, so time based questions like "what are the latest hits" reference 2024 information
### 3) It's important to be quite specific about what what the agent can and can't do.
### 4) Codex tried to write code that is very rigourous (classes etc.) - way too complex for a noob like me today. So need to ask lots of questions and fully understand


class SpotifyAPIError(RuntimeError):
    """Represents a Spotify Web API failure with an actionable message."""


# Used Codex to come up with the below SpotifyClient implementation.

class SpotifyClient:
    """Minimal Spotify Web API client using Client Credentials flow.

    Requires env vars: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
    """

    def __init__(self) -> None:
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.market = os.getenv("SPOTIFY_MARKET")
        self._token_override = os.getenv("SPOTIFY_ACCESS_TOKEN")
        if not self._token_override and (not self.client_id or not self.client_secret):
            raise RuntimeError(
                "Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in environment."
            )
        self._access_token: Optional[str] = None
        self._expires_at: float = 0

    def _refresh_token(self) -> None:
        if self._token_override:
            # Caller provided a pre-generated token (likely Authorization Code flow).
            self._access_token = self._token_override
            self._expires_at = float("inf")
            return

        token_url = "https://accounts.spotify.com/api/token"
        auth_header = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {"grant_type": "client_credentials"}
        resp = requests.post(token_url, data=data, headers=headers, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        self._access_token = payload["access_token"]
        # Refresh a minute before expiry for safety
        self._expires_at = time.time() + int(payload.get("expires_in", 3600)) - 60

    def _get_token(self) -> str:
        if self._token_override:
            return self._token_override

        if not self._access_token or time.time() >= self._expires_at:
            self._refresh_token()
        assert self._access_token is not None
        return self._access_token

    def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Accept": "application/json",
        }
        resp = requests.get(url, headers=headers, params=params or {}, timeout=20)
        if resp.status_code in (401, 403) and not self._token_override:
            # Token may have expired or scopes changed mid-run; refresh once.
            self._refresh_token()
            headers["Authorization"] = f"Bearer {self._get_token()}"
            resp = requests.get(url, headers=headers, params=params or {}, timeout=20)

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            message = self._extract_error_message(resp)
            raise SpotifyAPIError(message) from exc
        return resp.json()

    @staticmethod
    def _extract_error_message(resp: requests.Response) -> str:
        try:
            payload: Dict[str, Any] = resp.json()
        except ValueError:
            payload = {}
        status = resp.status_code
        message = None
        if isinstance(payload, dict):
            if "error" in payload:
                error_obj = payload["error"]
                if isinstance(error_obj, dict):
                    message = error_obj.get("message")
                elif isinstance(error_obj, str):
                    message = error_obj
        if not message:
            message = resp.text.strip() or "Unknown Spotify API error"

        if status == 403:
            message = (
                f"Spotify API returned 403 Forbidden: {message}. "
                "For development apps, ensure the requested resource is available in your market "
                "and consider using an Authorization Code token (set SPOTIFY_ACCESS_TOKEN)."
            )
        elif status == 404:
            message = (
                f"Spotify API returned 404 Not Found: {message}. "
                "This usually means one of the supplied IDs is invalid or unavailable in the current market."
            )
        else:
            message = f"Spotify API error {status}: {message}"
        return message

    def search_tracks(self, query: str, limit: int = 5) -> List[Dict]:
        params: Dict[str, Any] = {
            "q": query,
            "type": "track",
            "limit": max(1, min(limit, 20)),
        }
        if self.market:
            params["market"] = self.market
        result = self._get("https://api.spotify.com/v1/search", params=params)
        items = result.get("tracks", {}).get("items", [])
        tracks = []
        for t in items:
            tracks.append(
                {
                    "id": t.get("id"),
                    "name": t.get("name"),
                    "artists": ", ".join(a.get("name", "") for a in t.get("artists", [])),
                    "album": t.get("album", {}).get("name"),
                    "url": t.get("external_urls", {}).get("spotify"),
                    "preview_url": t.get("preview_url"),
                }
            )
        return tracks

    def audio_features(self, track_id: str) -> Dict:
        data = self._get("https://api.spotify.com/v1/audio-features", params={"ids": track_id})
        items = data.get("audio_features") if isinstance(data, dict) else None
        feature_obj: Optional[Dict[str, Any]] = None
        if isinstance(items, list) and items:
            feature_obj = items[0]
        if not feature_obj:
            raise SpotifyAPIError(
                f"Spotify did not return audio features for track '{track_id}'."
            )
        keys = [
            "tempo",
            "danceability",
            "energy",
            "valence",
            "acousticness",
            "instrumentalness",
            "liveness",
            "speechiness",
            "loudness",
            "time_signature",
        ]
        return {k: feature_obj.get(k) for k in keys}


spotify = None
today_iso = date.today().isoformat()


def _get_spotify() -> SpotifyClient:
    global spotify
    if spotify is None:
        spotify = SpotifyClient()
    return spotify


@function_tool
def spotify_search_tracks(query: str, limit: int = 5) -> List[Dict]:
    """Search Spotify for tracks matching a free-text query.

    Returns a list of tracks with id, name, artists, album, url, preview_url.
    """

    return _get_spotify().search_tracks(query=query, limit=limit)


def _normalize_track_id(value: str) -> str:
    value = value.strip()
    if value.startswith("spotify:track:"):
        return value.split(":")[-1]
    if "open.spotify.com/track" in value:
        segment = value.split("track/")[-1]
        return segment.split("?")[0]
    return value


@function_tool
def spotify_track_features(track_id: str) -> Dict:
    """Get audio features for a single track by ID (e.g., tempo, energy)."""

    clean_id = _normalize_track_id(track_id)
    return _get_spotify().audio_features(clean_id)


def get_todays_date():
    return time.strftime("%Y-%m-%d")


agent = Agent(
    name="Spotify DJ",
    instructions=(
        "You are a creative music assistant. Use the Spotify tools to search "
        "for tracks and analyze their audio features. When suggesting tracks, "
        "include title, main artist, and Spotify URL. Today's date is "
        f"{today_iso}. Use this when interpreting time-relative requests. "
        "Explain insights drawn from audio features when helpful."
    ),
    tools=[spotify_search_tracks, spotify_track_features],
)


async def main(prompt: str) -> None:
    result = await Runner.run(agent, input=prompt)
    print(result.final_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Spotify DJ agent against a custom prompt."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help=(
            "Question or instruction for the agent. If omitted, a default running-set "
            "prompt is used."
        ),
    )
    args = parser.parse_args()
    default_prompt = (
        "Find upbeat chill tracks suited for a calm evening. Share the Spotify links plus a quick note on the music style."
    )
    user_prompt = args.prompt or default_prompt
    asyncio.run(main(user_prompt))