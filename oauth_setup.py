"""
YouTube OAuth setup — PKCE + localhost redirect (no server needed).

Step 1 — generate URL:
    python3 oauth_setup.py --generate-url
    Opens Google auth. After approval, browser redirects to localhost:8090?code=XXX
    (redirect will fail visibly in browser — that's fine). Copy the full redirect URL
    or just the code= value from the address bar.

Step 2 — exchange code:
    python3 oauth_setup.py --exchange-code CODE
    Saves credentials/youtube_token.json. Done.
"""

import argparse
import base64
import hashlib
import json
import secrets
import sys
import urllib.parse
from pathlib import Path

import requests as http_requests
from google.oauth2.credentials import Credentials

PROJECT_ROOT = Path(__file__).parent
CLIENT_SECRET_PATH = PROJECT_ROOT / "credentials" / "client_secret.json"
TOKEN_PATH = PROJECT_ROOT / "credentials" / "youtube_token.json"
STATE_PATH = PROJECT_ROOT / "credentials" / ".oauth_state.json"

REDIRECT_URI = "http://localhost:8090"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]


def _load_client_secret() -> dict:
    if not CLIENT_SECRET_PATH.exists():
        print(f"ERROR: client_secret.json not found at {CLIENT_SECRET_PATH}")
        sys.exit(1)
    with open(CLIENT_SECRET_PATH) as f:
        data = json.load(f)
    return data.get("installed") or data.get("web")


def _pkce_pair() -> tuple:
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return code_verifier, code_challenge


def cmd_generate_url():
    client = _load_client_secret()
    client_id = client["client_id"]
    auth_uri = client.get("auth_uri", "https://accounts.google.com/o/oauth2/auth")
    token_uri = client.get("token_uri", "https://oauth2.googleapis.com/token")

    code_verifier, code_challenge = _pkce_pair()
    state = secrets.token_urlsafe(16)

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }

    auth_url = auth_uri + "?" + urllib.parse.urlencode(params)

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({
            "code_verifier": code_verifier,
            "state": state,
            "client_id": client_id,
            "client_secret": client["client_secret"],
            "token_uri": token_uri,
            "redirect_uri": REDIRECT_URI,
        }, f, indent=2)

    print(auth_url)


def cmd_exchange_code(auth_code: str):
    if not STATE_PATH.exists():
        print("ERROR: No saved OAuth state. Run --generate-url first.")
        sys.exit(1)

    with open(STATE_PATH) as f:
        state = json.load(f)

    resp = http_requests.post(
        state["token_uri"],
        data={
            "grant_type": "authorization_code",
            "code": auth_code.strip(),
            "redirect_uri": state["redirect_uri"],
            "client_id": state["client_id"],
            "client_secret": state["client_secret"],
            "code_verifier": state["code_verifier"],
        },
    )

    if not resp.ok:
        print(f"ERROR: Token exchange failed ({resp.status_code}): {resp.text}")
        sys.exit(1)

    token_data = resp.json()
    if "error" in token_data:
        print(f"ERROR: {token_data['error']}: {token_data.get('error_description', '')}")
        sys.exit(1)

    creds = Credentials(
        token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        token_uri=state["token_uri"],
        client_id=state["client_id"],
        client_secret=state["client_secret"],
        scopes=SCOPES,
    )

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())

    STATE_PATH.unlink(missing_ok=True)

    print(f"Token saved to: {TOKEN_PATH}")
    print("YouTube OAuth complete")


def main():
    parser = argparse.ArgumentParser(description="YouTube OAuth setup")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate-url", action="store_true")
    group.add_argument("--exchange-code", metavar="CODE")
    args = parser.parse_args()

    if args.generate_url:
        cmd_generate_url()
    else:
        cmd_exchange_code(args.exchange_code)


if __name__ == "__main__":
    main()
