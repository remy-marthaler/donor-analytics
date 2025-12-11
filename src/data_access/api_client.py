import os
import time
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import streamlit as st

from src.core.config import FBOX_BASE
from src.data_access.columns import ALL_COLUMNS


class ApiClient:
    """Real FundraisingBox API client with caching and pagination."""

    def __init__(self, timeout: float = 30.0):
        # Load API token from environment
        token = os.getenv("FBOX_API_KEY")

        # Fail early if no API key is configured
        if not token:
            raise RuntimeError("Missing FBOX_API_KEY in .env")

        # BasicAuth: token as username, dummy password
        self.auth = HTTPBasicAuth(token, "X")
        self.timeout = timeout

        # Reuse a persistent session for better performance
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Internal GET with retries (handles rate limits + temporary errors)
    # ------------------------------------------------------------------
    def _get(self, path: str, params=None):
        url = f"{FBOX_BASE}{path}"
        params = params or {}

        for attempt in range(5):
            try:
                # Perform the GET request with auth and timeout
                r = self.session.get(url, params=params, auth=self.auth, timeout=self.timeout)

                # Retry on rate-limiting or transient server errors
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 10))  # exponential backoff
                    continue

                # Raise readable errors for non-OK responses
                if not r.ok:
                    try:
                        msg = r.json().get("error") or r.json().get("message")
                    except Exception:
                        msg = r.text
                    raise RuntimeError(f"FBOX error {r.status_code}: {msg}")

                # Success: return parsed JSON
                return r.json()

            except requests.RequestException:
                # Network failures: wait and retry
                time.sleep(min(2 ** attempt, 10))

        # All retries failed
        raise RuntimeError("FundraisingBox request failed after retries")

    # ------------------------------------------------------------------
    # Pagination for donations (yields each donation row)
    # ------------------------------------------------------------------
    def _paginate_donations(self, params):
        page = 1
        while True:
            payload = dict(params or {})
            payload["page"] = page

            # Fetch page of results
            data = self._get("/donations.json", payload)
            rows = data.get("data", []) or []
            has_more = bool(data.get("hasMore"))

            # Stop if empty
            if not rows:
                break

            # Yield rows one by one
            for row in rows:
                yield row

            # No more pages → stop
            if not has_more:
                break

            page += 1

    # ------------------------------------------------------------------
    # STREAMLIT-CACHE SAFE WRAPPERS
    # ------------------------------------------------------------------

    @staticmethod
    @st.cache_data(show_spinner=True, ttl=60 * 60 * 6)
    def _cached_donations(_self, since, until):
        # Build date filters for API
        params = {}

        if since:
            params["date_min"] = str(since)
        if until:
            params["date_max"] = str(until)

        # Fetch all donations via pagination
        raw_rows = list(_self._paginate_donations(params))

        # No data → return empty schema
        if not raw_rows:
            return pd.DataFrame(columns=ALL_COLUMNS)

        # Flatten JSON into table
        df = pd.json_normalize(raw_rows)

        # Map FBOX fields to internal column names
        mapping = {
            "fb_person_id": "Kontakt-ID",
            "received_at": "Getätigt am Datum",
            "amount": "Betrag",
            "fb_source_id": "Quelle",
            "source_name": "Zahlungsweise",
            "fb_project_id": "Projekt"
        }
        df = df.rename(columns=mapping)

        # Ensure all required columns exist, even if missing from API
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Return clean dataframe with correct order
        return df[ALL_COLUMNS].reset_index(drop=True)

    @staticmethod
    @st.cache_data(show_spinner=True, ttl=60 * 60 * 6)
    def _cached_donors(_self):
        # Fetch all persons via pagination
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

        # No data → empty schema
        if not persons:
            return pd.DataFrame(columns=ALL_COLUMNS)

        # Flatten JSON into table
        df = pd.json_normalize(persons)

        # Hydrate only the Kontakt-ID column (others not provided by endpoint)
        out = pd.DataFrame(columns=ALL_COLUMNS)

        if "id" in df.columns:
            out["Kontakt-ID"] = df["id"]

        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # PUBLIC METHODS (API-compatible)
    # ------------------------------------------------------------------
    def get_donations(self, since=None, until=None):
        # Use cached donation results
        return ApiClient._cached_donations(self, since, until)

    def get_donors(self):
        # Use cached donor results
        return ApiClient._cached_donors(self)
