
"""
calendar_utils.py

Trading-calendar helpers built on exchange_calendars.

We default to XNYS sessions, which is a practical approximation for US equities.

If you trade ADRs or non-US listings, you can parameterize the calendar name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

try:
    import exchange_calendars as ecals
except Exception as e:  # pragma: no cover
    ecals = None


class CalendarError(RuntimeError):
    pass


@dataclass(frozen=True)
class TradingCalendar:
    name: str = "XNYS"

    def _get_cal(self):
        if ecals is None:
            raise CalendarError("exchange_calendars is not installed. Install with: pip install exchange_calendars")
        try:
            return ecals.get_calendar(self.name)
        except Exception as e:
            raise CalendarError(f"Failed to load exchange calendar '{self.name}': {e}")

    def sessions_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        cal = self._get_cal()
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        # sessions_in_range is inclusive
        return cal.sessions_in_range(start, end)

    def next_session(self, session: pd.Timestamp) -> pd.Timestamp:
        cal = self._get_cal()
        session = pd.Timestamp(session).normalize()
        nxt = cal.next_session(session)
        return pd.Timestamp(nxt).normalize()

    def _snap_to_valid_session(self, session: pd.Timestamp, direction: str = "forward") -> pd.Timestamp:
        """
        If session is not a valid trading day, snap to nearest valid session.
        direction: 'forward' or 'backward'
        """
        cal = self._get_cal()
        session = pd.Timestamp(session).normalize()

        # Check if already valid
        sessions = cal.sessions_in_range(session, session)
        if len(sessions) == 1:
            return session

        # Not a valid trading day - search in a 10-day window
        if direction == "forward":
            search_end = session + pd.Timedelta(days=10)
            nearby = cal.sessions_in_range(session, search_end)
        else:
            search_start = session - pd.Timedelta(days=10)
            nearby = cal.sessions_in_range(search_start, session)

        if len(nearby) == 0:
            raise CalendarError(f"No valid trading sessions near {session.date()}")

        if direction == "forward":
            return pd.Timestamp(nearby[0]).normalize()
        else:
            return pd.Timestamp(nearby[-1]).normalize()

    def add_sessions(self, session: pd.Timestamp, n: int) -> pd.Timestamp:
        """
        Add n trading sessions to 'session'.
        If n=0 returns the same session (or next valid session if input is not a trading day).

        If 'session' is not a valid trading day (weekend/holiday), we first snap to
        the next valid trading session before adding n sessions.
        """
        cal = self._get_cal()
        session = pd.Timestamp(session).normalize()

        # Snap to valid session if needed
        session = self._snap_to_valid_session(session, direction="forward")

        if n == 0:
            return session

        # To add n sessions, we can step iteratively using next_session for robustness
        cur = session
        step = 1 if n > 0 else -1
        for _ in range(abs(n)):
            if step > 0:
                cur = pd.Timestamp(cal.next_session(cur)).normalize()
            else:
                cur = pd.Timestamp(cal.previous_session(cur)).normalize()
        return cur
