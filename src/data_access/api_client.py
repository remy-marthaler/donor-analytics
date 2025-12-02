import os
import time
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import streamlit as st

from src.data_access.columns import ALL_COLUMNS

FBOX_BASE = "https://api.fundraisingbox.com/v1"


class ApiClient:
    """Real FundraisingBox API client with caching and pagination."""

    def __init__(self, timeout: float = 30.0):
        token = (
            os.getenv("FBOX_API_KEY")
            or os.getenv("FBOX_TOKEN")
            or os.getenv("FUNDRAISINGBOX_TOKEN")
        )

        if not token:
            raise RuntimeError("Missing FBOX_API_KEY in .env")

        self.auth = HTTPBasicAuth(token, "X")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Internal GET with retries
    # ------------------------------------------------------------------
    def _get(self, path: str, params=None):
        url = f"{FBOX_BASE}{path}"
        params = params or {}

        for attempt in range(5):
            try:
                r = self.session.get(url, params=params, auth=self.auth, timeout=self.timeout)

                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 10))
                    continue

                if not r.ok:
                    try:
                        msg = r.json().get("error") or r.json().get("message")
                    except Exception:
                        msg = r.text
                    raise RuntimeError(f"FBOX error {r.status_code}: {msg}")

                return r.json()

            except requests.RequestException:
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError("FundraisingBox request failed after retries")

    # ------------------------------------------------------------------
    # Pagination for donations
    # ------------------------------------------------------------------
    def _paginate_donations(self, params):
        page = 1
        while True:
            payload = dict(params or {})
            payload["page"] = page

            data = self._get("/donations.json", payload)
            rows = data.get("data", []) or []
            has_more = bool(data.get("hasMore"))

            if not rows:
                break

            for row in rows:
                yield row

            if not has_more:
                break

            page += 1

    # ------------------------------------------------------------------
    # STREAMLIT-CACHE SAFE WRAPPERS
    # ------------------------------------------------------------------

    @staticmethod
    @st.cache_data(show_spinner=True, ttl=60 * 60 * 6)
    def _cached_donations(_self, since, until):
        params = {}

        if since:
            params["date_min"] = str(since)
        if until:
            params["date_max"] = str(until)

        raw_rows = list(_self._paginate_donations(params))

        if not raw_rows:
            return pd.DataFrame(columns=ALL_COLUMNS)

        df = pd.json_normalize(raw_rows)

        # map real FBOX fields → expected dataframe columns
        mapping = {
            "fb_person_id": "Kontakt-ID",
            "received_at": "Getätigt am Datum",
            "amount": "Betrag",
        }

        df = df.rename(columns=mapping)

        # ensure all required columns exist
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = None

        return df[ALL_COLUMNS].reset_index(drop=True)

    @staticmethod
    @st.cache_data(show_spinner=True, ttl=60 * 60 * 6)
    def _cached_donors(_self):
        persons = []
        page = 1

        while True:
            data = _self._get("/persons.json", {"page": page, "per_page": 1000})
            rows = data.get("data", []) or []
            has_more = bool(data.get("hasMore"))

            if not rows:
                break

            persons.extend(rows)

            if not has_more:
                break

            page += 1

        if not persons:
            return pd.DataFrame(columns=ALL_COLUMNS)

        df = pd.json_normalize(persons)

        out = pd.DataFrame(columns=ALL_COLUMNS)

        if "id" in df.columns:
            out["Kontakt-ID"] = df["id"]

        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # PUBLIC METHODS (API-compatible)
    # ------------------------------------------------------------------
    def get_donations(self, since=None, until=None):
        return ApiClient._cached_donations(self, since, until)

    def get_donors(self):
        return ApiClient._cached_donors(self)
