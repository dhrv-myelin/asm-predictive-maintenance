# DataHandler - Cycle vs Time-based Indexing

```python
class DataHandler:
    def __init__(self, config: dict, target_name: str):
        self.config = config
        self.target_name = target_name
        self.format = config.get("format", "cycle")  # "cycle" or "time"

        self.history_window = config["history_window"]
        self.stride = config.get("stride", 0)
        self.prediction_window = config["prediction_window"]
        self.required_features = config["required_features"]

        # Buffers based on format
        if self.format == "cycle":
            self._cycle_buffer: dict[tuple[date, int], dict] = {}
            self._max_cycle_per_day: dict[date, int] = {}
        else:
            self._time_bucket_seconds = config.get("bucket_duration", 60)
            self._time_buffer: dict[int, dict] = {}  # bucket_id -> dict

        self.df = pd.DataFrame(columns=self.required_features)

    def _get_buffer_key(self, ts, cycle_count) -> tuple:
        if self.format == "cycle":
            day = ts.date()
            return (day, cycle_count)
        else:
            bucket_id = int(ts.timestamp() // self._time_bucket_seconds)
            return bucket_id

    def _is_complete(self, key) -> bool:
        if self.format == "cycle":
            day, cycle = key
            max_for_day = self._max_cycle_per_day.get(day, 0)
            return cycle < max_for_day
        else:
            # Time-based: bucket is complete when newer bucket arrives
            current_bucket = self._get_current_bucket()
            return key < current_bucket

    def _get_current_bucket(self) -> int:
        return int(pd.Timestamp.now().timestamp() // self._time_bucket_seconds)

    def ingest(self, rows) -> pd.DataFrame | None:
        if not rows:
            return None

        for r in rows:
            ts = pd.to_datetime(r.timestamp)
            nmn = f"{r.station_name}__{r.metric_name}"
            cycle_count = r.cycle_count

            key = self._get_buffer_key(ts, cycle_count)

            if key not in self._buffer:
                self._buffer[key] = {"timestamp": ts}
            self._buffer[key][nmn] = r.value

            # Track max for cycle-based
            if self.format == "cycle":
                day = ts.date()
                if day not in self._max_cycle_per_day or cycle_count > self._max_cycle_per_day[day]:
                    self._max_cycle_per_day[day] = cycle_count

        self._flush_completed()
        return self.df if not self.df.empty else None

    def _flush_completed(self):
        keys_to_flush = [k for k in self._buffer if self._is_complete(k)]
        keys_to_flush.sort()

        for key in keys_to_flush:
            row_dict = self._buffer.pop(key)
            self._append_row(row_dict)

    def flush_remaining(self):
        # For cycle: flush all
        # For time: may want to flush only buckets older than N minutes
        for key in sorted(self._buffer.keys()):
            self._append_row(self._buffer.pop(key))

    # fetch_next_window and fetch_train_data stay the same
    # (operate on assembled df, independent of indexing strategy)
```

---

## Key differences

| Aspect | Cycle | Time |
|--------|-------|------|
| Buffer key | `(date, cycle)` | `timestamp // bucket_duration` |
| Completeness | `cycle < max_for_day` | `bucket < current_bucket` |
| Flush trigger | New cycle arrives | New time bucket starts |
| Incomplete data | Stays until explicitly flushed | Stays until timeout or explicit flush |