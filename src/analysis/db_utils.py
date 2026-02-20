from sqlalchemy import text
from datetime import timedelta

class DBUtils:
    def __init__(self, session_factory):
        self._sf = session_factory
        self.last_written_timestamps = []
    
    def fetch_data(self, start_timestamp, end_timestamp):
        sql = text("""
            SELECT timestamp, station_name, metric_name, value
            FROM process_metrics
            WHERE timestamp >= :start_ts
            AND timestamp <= :end_ts
            ORDER BY timestamp ASC
        """)

        with self._sf() as s:
            rows = s.execute(sql, {
                "start_ts": start_timestamp,
                "end_ts": end_timestamp
            }).fetchall()

        if not rows:
            return []

        return rows


    def insert_results(self, last_timestamp, values):

        # remove old predictions
        for ts in self.last_written_timestamps:
            self._remove_from_db(ts)

        self.last_written_timestamps = []

        curr_ts = last_timestamp

        for v in values:
            curr_ts = curr_ts + timedelta(seconds=1)
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
