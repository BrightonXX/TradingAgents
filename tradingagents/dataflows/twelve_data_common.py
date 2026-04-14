import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

from .alpha_vantage_common import AlphaVantageRateLimitError

API_BASE_URL = "https://api.twelvedata.com"

# Rate limit: 8 requests per minute for free tier
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 8.0  # seconds between requests


def _rate_limit_throttle():
    """Ensure minimum interval between API requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


class TwelveDataRateLimitError(AlphaVantageRateLimitError):
    """Twelve Data rate limit error. Inherits from AlphaVantageRateLimitError
    so the existing fallback mechanism in interface.py catches it automatically."""
    pass


def get_api_key() -> str:
    """Retrieve the API key for Twelve Data from environment variables."""
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY environment variable is not set.")
    return api_key


def _make_api_request(endpoint: str, params: dict = None) -> dict:
    """Make an API request to Twelve Data.

    Raises:
        TwelveDataRateLimitError: When rate limit is exceeded or endpoint not found.
    """
    if params is None:
        params = {}
    params["apikey"] = get_api_key()

    _rate_limit_throttle()

    url = f"{API_BASE_URL}/{endpoint}"
    response = requests.get(url, params=params)

    # 404 means endpoint not available on this plan
    if response.status_code == 404:
        raise TwelveDataRateLimitError(
            f"Twelve Data endpoint '{endpoint}' not available. Falling back."
        )

    response.raise_for_status()

    data = response.json()

    # Twelve Data returns errors in the JSON body
    if isinstance(data, dict):
        status = data.get("status")
        if status == "error":
            message = data.get("message", "")
            if "rate limit" in message.lower() or "api credits" in message.lower() or data.get("code") == 429:
                raise TwelveDataRateLimitError(
                    f"Twelve Data rate limit exceeded: {message}"
                )
            # Other API errors - also raise to trigger fallback
            raise TwelveDataRateLimitError(f"Twelve Data API error: {message}")

    return data


def _calc_start_date(curr_date: str, look_back_days: int) -> str:
    """Calculate start date from curr_date minus look_back_days."""
    dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start = dt - timedelta(days=int(look_back_days * 1.5))  # extra buffer for weekends/holidays
    return start.strftime("%Y-%m-%d")


def _filter_csv_by_date_range(csv_data: str, start_date: str, end_date: str) -> str:
    """Filter CSV data to include only rows within the specified date range."""
    if not csv_data or csv_data.strip() == "":
        return csv_data
    try:
        df = pd.read_csv(StringIO(csv_data))
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]
        return filtered.to_csv(index=False)
    except Exception:
        return csv_data
