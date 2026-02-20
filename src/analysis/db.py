from sqlalchemy import create_engine, text
import pandas as pd
from datetime import timedelta

# --------------------------------------------------
# writing results
# --------------------------------------------------
def insert_results(self, last_timestamp, values):
    """
    values: iterable of predictions
    """

    # remove old predictions
    for ts in self.last_written_timestamps:
        self._remove_from_db(ts)

    self.last_written_timestamps = []

    curr_ts = last_timestamp

    for v in values:
        curr_ts = curr_ts + timedelta(seconds=1)  # or cycle-based logic
        self._write_db(curr_ts, v)
        self.last_written_timestamps.append(curr_ts)

def _remove_from_db(self, timestamp):
    sql = text("""
        DELETE FROM analysis_results
        WHERE timestamp = :ts
            AND target = :target
    """)
    with self._sf() as s:
        s.execute(sql, {"ts": timestamp, "target": self.target_name})
        s.commit()

def _write_db(self, timestamp, value):
    sql = text("""
        INSERT INTO analysis_results (timestamp, target, value)
        VALUES (:ts, :target, :val)
    """)
    with self._sf() as s:
        s.execute(sql, {
            "ts": timestamp,
            "target": self.target_name,
            "val": float(value)
        })
        s.commit()