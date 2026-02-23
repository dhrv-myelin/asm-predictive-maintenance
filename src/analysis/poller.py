import time
import threading
from sqlalchemy import text
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger(__name__)

class DBPoller:
    def __init__(self, session_factory, poll_interval=5):
        self._sf = session_factory
        self.poll_interval = poll_interval
        self._last_ts = None
        self.if_stop = threading.Event()

    def start(self, target_func):
        t = threading.Thread(target=target_func, daemon=True)
        t.start()

    def stop(self):
        self.if_stop.set()

    def poll(self):
        since = self._last_ts or (datetime.now() - timedelta(minutes=5)) # Add for UTC time stamp: timezone.utc
        # since = self._last_ts or datetime.fromisoformat("2024-01-24T03:00:00+00:00")
        # print("since : ",since)

        sql = text("""
            SELECT timestamp, station_name, metric_name, value
            FROM process_metrics
            WHERE timestamp > :since
            ORDER BY timestamp ASC
        """)

        with self._sf() as s:
            rows = s.execute(sql, {"since": since}).fetchall()

        if not rows:
            return None

        self._last_ts = rows[-1].timestamp
        return rows
