"""
scheduler/poller.py
─────────────────────
DB polling trigger for the inference system.

Watches the process_metrics TimescaleDB hypertable for new rows.
When enough new cycles have accumulated (min_new_cycles), fires an
orchestrator run — debounced so it can't run more often than
min_run_interval_seconds.

Cycle detection: uses the system__count metric as the cycle boundary signal.
Each unique value of system__count that arrived after last_seen_ts is one
new cycle. This is more reliable than counting raw rows because each cycle
produces many rows in the long-format table.

Runs as a daemon thread started from the FastAPI lifespan event.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


class DBPoller:
    """
    Parameters
    ----------
    session_factory         : callable → context-manager yielding SQLAlchemy Session
    orchestrator            : InferenceOrchestrator instance
    poll_interval_seconds   : how often to query the DB (default 30s)
    min_new_cycles          : number of new cycles needed to trigger a run (default 5)
    min_run_interval_seconds: minimum gap between consecutive runs (default 120s)
    """

    def __init__(
        self,
        session_factory         ,
        orchestrator            ,
        poll_interval_seconds   : int = 30,
        min_new_cycles          : int = 5,
        min_run_interval_seconds: int = 120,
    ) -> None:
        self._sf               = session_factory
        self._orchestrator     = orchestrator
        self._poll_interval    = poll_interval_seconds
        self._min_new_cycles   = min_new_cycles
        self._min_run_interval = min_run_interval_seconds

        # State tracking
        self._last_seen_ts : Optional[datetime] = None
        self._last_run_at  : Optional[datetime] = None

        self._stop_event = threading.Event()
        self._thread     : Optional[threading.Thread] = None

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info(
            "DBPoller starting (interval=%ds, min_cycles=%d, debounce=%ds)",
            self._poll_interval, self._min_new_cycles, self._min_run_interval,
        )
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="db-poller"
        )
        self._thread.start()

    def stop(self, timeout: int = 10) -> None:
        logger.info("DBPoller stopping…")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    # ──────────────────────────────────────────────────────────────────────
    # Poll loop
    # ──────────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.error("DBPoller tick error: %s", exc, exc_info=True)
            self._stop_event.wait(timeout=self._poll_interval)

    def _tick(self) -> None:
        now    = datetime.now(timezone.utc)
        # On first tick, look back 1 hour to avoid triggering immediately on startup
        since  = self._last_seen_ts or (now - timedelta(hours=1))

        new_cycles, latest_ts = self._count_new_cycles(since)
        logger.debug("DBPoller: %d new cycles since %s", new_cycles, since)

        if new_cycles < self._min_new_cycles:
            return

        # Debounce
        if self._last_run_at:
            elapsed = (now - self._last_run_at).total_seconds()
            if elapsed < self._min_run_interval:
                logger.debug(
                    "DBPoller: debounced (last run %.0fs ago, need %ds)",
                    elapsed, self._min_run_interval,
                )
                return

        # Trigger inference
        logger.info(
            "DBPoller: %d new cycles detected — triggering inference run", new_cycles
        )
        self._last_seen_ts = latest_ts
        self._last_run_at  = now

        try:
            run_id = self._orchestrator.run(triggered_by="db_poll")
            logger.info("DBPoller: inference run %s triggered", run_id)
        except Exception as exc:
            logger.error("DBPoller: failed to trigger inference run: %s", exc, exc_info=True)

    # ──────────────────────────────────────────────────────────────────────
    # DB query
    # ──────────────────────────────────────────────────────────────────────

    def _count_new_cycles(
        self, since: datetime
    ) -> tuple[int, Optional[datetime]]:
        """
        Count distinct system__count values that arrived after `since`.
        This equals the number of new complete manufacturing cycles.

        Returns (n_new_cycles, latest_timestamp).
        Uses TimescaleDB's hypertable chunk exclusion via the timestamp filter.
        """
        sql = text("""
            SELECT
                COUNT(DISTINCT value)       AS new_cycles,
                MAX(timestamp)              AS latest_ts
            FROM process_metrics
            WHERE station_name = 'system'
              AND metric_name  = 'count'
              AND timestamp    > :since
        """)

        try:
            with self._sf() as session:
                row = session.execute(sql, {"since": since}).one()
            n_cycles  = int(row.new_cycles or 0)
            latest_ts = row.latest_ts
            return n_cycles, latest_ts
        except Exception as exc:
            logger.error("DBPoller: cycle count query failed: %s", exc)
            return 0, None


class ManualRetrain:
    """
    Exposes a manual retrain trigger decoupled from the polling loop.
    Called directly by the API endpoint POST /inference/retrain/{station}/{metric}.
    """

    def __init__(self, orchestrator) -> None:
        self._orchestrator = orchestrator

    def trigger(self, station: str, metric: str) -> Optional[str]:
        """
        Force retrain for one station+metric.
        Returns the MLflow run_id string, or None on failure.
        """
        logger.info("Manual retrain: %s/%s", station, metric)
        return self._orchestrator.retrain_manual(station, metric)
