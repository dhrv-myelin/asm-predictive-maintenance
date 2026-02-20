# poller.py

import time
import threading
from sqlalchemy import text
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class DBPoller:
    def __init__(self, session_factory, dataloaders, poll_interval=5):
        self._sf = session_factory
        self._dataloaders = dataloaders   # list of DataHandler instances
        self.poll_interval = poll_interval
        self._last_ts = None
        self.if_stop = threading.Event()

    def start(self,target_func):
        t = threading.Thread(target=target_func, daemon=True)
        t.start()

    def stop(self):
        self.if_stop.set()

    # def _loop(self):
    #     while not self.if_stop.is_set():
    #         try:
    #             self._poll()
    #         except Exception as e:
    #             logger.exception("Poller error: %s", e)
    #         time.sleep(self._poll_interval)

    def poll(self):
        since = self._last_ts or datetime.now(timezone.utc)

        sql = text("""
            SELECT timestamp, station_name, metric_name, value
            FROM process_metrics
            WHERE timestamp > :since
            ORDER BY timestamp ASC
        """)

        with self._sf() as s:
            rows = s.execute(sql, {"since": since}).fetchall()

        if not rows:
            return

        self._last_ts = rows[-1].timestamp

        # broadcast
        # for dl in self._dataloaders:
        #     dl.ingest(rows)
        return rows
