from datetime import timedelta

import pandas as pd
from polars import pl
from sqlalchemy import text


class DBUtils:
    def __init__(self, session_factory):
        self._sf = session_factory

    # --------------------------------------------------
    # Existing: fetch process_metrics rows
    # --------------------------------------------------

    def fetch_data(self, start_timestamp, end_timestamp):
        sql = text("""
            SELECT timestamp, station_name, metric_name, value, cycle_count
            FROM process_metrics
            WHERE timestamp >= :start_ts
            AND timestamp <= :end_ts
            ORDER BY timestamp ASC
        """)

        with self._sf() as s:
            rows = s.execute(
                sql,
                {
                    "start_ts": start_timestamp,
                    "end_ts": end_timestamp,
                },
            ).fetchall()

        if not rows:
            print(
                f"[ERROR] No data found between {start_timestamp} and {end_timestamp}"
            )
            return []

        return rows

    # --------------------------------------------------
    # NEW: fetch all process_metrics (for stats pipeline)
    # --------------------------------------------------

    def fetch_all_process_metrics(self) -> pd.DataFrame:
        """
        Pull the entire process_metrics table and return it as a DataFrame.
        Used by the stats pipeline which operates over the full 36-day history.
        """
        sql = text("""
            SELECT timestamp, station_name, metric_name, value, cycle_count
            FROM process_metrics
            ORDER BY timestamp ASC
        """)

        with self._sf() as s:
            rows = s.execute(sql).fetchall()

        if not rows:
            print("[ERROR] process_metrics table is empty")
            return pd.DataFrame(
                columns=["timestamp", "station_name", "metric_name", "value"]
            )

        df = pd.DataFrame(
            rows, columns=["timestamp", "station_name", "metric_name", "value"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    # --------------------------------------------------
    # NEW: fetch baseline table → dict for stats pipeline
    # --------------------------------------------------

    def fetch_baseline(self) -> dict[str, tuple[float, float]]:
        """
        Pull the baseline table and return a dict keyed by metric_name:
            { "dispensing_time": (mean, std), ... }

        Expected columns in the baseline table:
            metric_name  TEXT
            mean         FLOAT
            std_dev      FLOAT

        This is consumed directly by baseline_stats.load_baseline() logic
        and stats_model_2._resolve_baseline().
        """
        sql = text("""
            SELECT metric_name, mean, std_dev
            FROM baseline_metrics
        """)

        with self._sf() as s:
            rows = s.execute(sql).fetchall()

        if not rows:
            print(
                "[WARNING] baseline table is empty — stats pipeline will use local fallbacks"
            )
            return {}

        baseline = {}
        for row in rows:
            name = str(row[0]).strip()
            try:
                mean = float(row[1]) if row[1] is not None else None
                std = float(row[2]) if row[2] is not None else None
                if mean is not None and std is not None:
                    baseline[name] = (mean, max(std, 1e-6))
                else:
                    print(
                        f"[WARNING] Null mean/std for '{name}' in baseline — skipping"
                    )
            except Exception as e:
                print(f"[WARNING] Could not parse baseline row for '{name}': {e}")

        print(f"[INFO] Loaded {len(baseline)} baseline entries from DB")
        return baseline

    # --------------------------------------------------
    # NEW: insert stats pattern results → patterns table
    # --------------------------------------------------

    def insert_patterns(self, patterns_df: pd.DataFrame) -> None:
        """
        Bulk-insert a patterns DataFrame produced by stats_model_2.run_pattern_pipeline()
        into the `patterns` table.

        Expected DataFrame columns (matches run_pattern_pipeline output):
            actual_timestamp     datetime (UTC)
            predicted_timestamp  datetime (UTC)
            predicted_value      float
            station_name         str
            metric_name          str
            model_name           str   — e.g. "stats_pattern_detector::Random spikes"
        """
        if patterns_df is None or patterns_df.empty:
            print("[INFO] No patterns to insert")
            return

        sql = text("""
            INSERT INTO model_predictions
                (actual_timestamp, predicted_timestamp, predicted_value,
                 station_name, metric_name, model_name)
            VALUES
                (:actual_timestamp, :predicted_timestamp, :predicted_value,
                 :station_name, :metric_name, :model_name)
        """)

        records = patterns_df[
            [
                "actual_timestamp",
                "predicted_timestamp",
                "predicted_value",
                "station_name",
                "metric_name",
                "model_name",
            ]
        ].to_dict(orient="records")

        with self._sf() as s:
            s.execute(sql, records)
            s.commit()

        print(f"[INFO] Inserted {len(records)} pattern rows into `patterns` table")

    # --------------------------------------------------
    # Existing: insert model prediction results
    # --------------------------------------------------

    def insert_results(
        self, last_timestamp, values, station_name, metric_name, model_name
    ):
        curr_ts = last_timestamp

        for v in values:
            curr_ts = curr_ts + timedelta(seconds=v)
            self._write_db(
                actual_timestamp=last_timestamp,
                predicted_timestamp=curr_ts,
                predicted_value=v,
                station_name=station_name,
                metric_name=metric_name,
                model_name=model_name,
            )

    def _write_db(
        self,
        actual_timestamp,
        predicted_timestamp,
        predicted_value,
        station_name,
        metric_name,
        model_name,
    ):
        sql = text("""
            INSERT INTO model_predictions
                (actual_timestamp, predicted_timestamp, predicted_value,
                 station_name, metric_name, model_name)
            VALUES
                (:actual_timestamp, :predicted_timestamp, :predicted_value,
                 :station_name, :metric_name, :model_name)
        """)
        with self._sf() as s:
            s.execute(
                sql,
                {
                    "actual_timestamp": actual_timestamp,
                    "predicted_timestamp": predicted_timestamp,
                    "predicted_value": predicted_value,
                    "station_name": station_name,
                    "metric_name": metric_name,
                    "model_name": model_name,
                },
            )
            s.commit()
