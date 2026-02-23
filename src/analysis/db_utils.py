from sqlalchemy import text
from datetime import timedelta

class DBUtils:
    def __init__(self, session_factory):
        self._sf = session_factory
    
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


    def insert_results(self, last_timestamp, values, station_name, metric_name, model_name):

        curr_ts = last_timestamp

        for v in values:
            curr_ts = curr_ts + timedelta(seconds=1)
            self._write_db(
                actual_timestamp = last_timestamp,
                predicted_timestamp = curr_ts,
                predicted_value = v,
                station_name = self.station_name,
                metric_name = self.metric_name,
                model_name = self.model_name
            )

    def _write_db(self,actual_timestamp, predicted_timestamp, predicted_value, station_name, metric_name, model_name):
        sql = text("""
            INSERT INTO model_predictions (id, actual_timestamp, predicted_timestamp, predicted_value, station_name, metric_name, model_name)
            VALUES (:id, :actual_timestamp, :predicted_timestamp, :predicted_value, :station_name, :metric_name, :model_name)
        """)
        with self._sf() as s:
            s.execute(sql, {
                "id": None,
                "actual_timestamp": actual_timestamp,
                "predicted_timestamp": predicted_timestamp,
                "predicted_value": predicted_value,
                "station_name": station_name,
                "metric_name": metric_name,
                "model_name": model_name
            })
            s.commit()
