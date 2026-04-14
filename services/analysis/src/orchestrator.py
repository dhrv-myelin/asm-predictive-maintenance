import argparse
import logging
import os
import threading
import time
from datetime import datetime, timezone

import mlflow
import yaml
from data_handler import DataHandler
from db_utils import DBUtils
from matplotlib.pylab import rint
from model import Model
from poller import DBPoller
from sklearn.preprocessing import RobustScaler
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from zoneinfo import ZoneInfo
from typing import Optional

# Stats pipeline imports
from baseline_stats import analyse_metric, build_config, ALLOWED_METRICS
from stats_model_2 import run_pattern_pipeline, print_summary

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# Main orchestrator loop
# --------------------------------------------------


def loop(poller, data_handlers, models, db_util):
    print("[DEBUG] Initialise Data Handlers : ", data_handlers)
    print("[DEBUG] Initialise Models : ", models)
    curr_ts_map = {name: None for name in data_handlers}

    inference_counter = 0

    while not poller.if_stop.is_set():
        try:
            print("[DEBUG] Polling for data...")
            rows = poller.poll()
            print("[DEBUG] Rows : ", rows)
            if not rows:
                time.sleep(poller.poll_interval)
                continue

            for name, handler in data_handlers.items():

                handler.ingest(rows)

                window = handler.fetch_next_window(
                    curr_first_timestamp=curr_ts_map[name], for_training=False
                )

                if window is not None:
                    inference_counter += 1
                    print(f"[DEBUG] Inference number : {inference_counter}")
                    model = models[name]

                    # inference
                    preds = model.real_time_inference(window)
                    preds = [preds[-1]]

                    # write
                    last_ts = window.iloc[-1]["timestamp"]
                    db_util.insert_results(
                        last_timestamp=last_ts,
                        values=preds,
                        station_name=handler.target_name.split("__")[0],
                        metric_name=handler.target_name.split("__")[1],
                        model_name=model.model_name,
                    )

                    # move window
                    curr_ts_map[name] = window.iloc[0]["timestamp"]

        except Exception as e:
            logger.exception("Orchestrator error: %s", e)

        time.sleep(poller.poll_interval)


def inference_loop(data_handler, model, db_util):
    has_timestamp = "timestamp" in data_handler.df.columns
    curr_first_timestamp = None
    curr_idx = 0

    while True:
        if has_timestamp:
            X = data_handler.fetch_next_window(curr_first_timestamp, for_training=False)
        else:
            start = curr_idx
            end = start + data_handler.history_window
            if end > len(data_handler.df):
                print("❌ No more windows")
                break
            X = data_handler.df.iloc[start:end]

        if X is None or (hasattr(X, "empty") and X.empty):
            print("❌ Not enough data for window")
            time.sleep(1)
            continue

        if has_timestamp:
            curr_first_timestamp = X.iloc[0]["timestamp"]
        else:
            curr_idx += data_handler.stride

        print("✅ Window shape:", X.shape)
        preds = model.real_time_inference(X)
        preds = [preds[-1]]

        last_ts = (
            X.iloc[-1]["timestamp"] if has_timestamp else data_handler.last_timestamp
        )
        print("[DEBUG] Last timestamp in window:", last_ts)
        db_util.insert_results(
            last_timestamp=last_ts,
            values=preds,
            station_name=data_handler.target_name.split("__")[0],
            metric_name=data_handler.target_name.split("__")[1],
            model_name=model.model_name,
        )

        time.sleep(0.5)


def infer_from_archive(start_ts, end_ts, data_handlers, models, db_util):
    rows = db_util.fetch_data(start_ts, end_ts)
    print(f"[DEBUG] Last UTC timestamp in Rows fetched : {rows[-1].timestamp}")
    if not rows:
        return

    def start(target_func):
        t = threading.Thread(target=target_func)
        t.start()
        return t

    threads = []
    for name, handler in data_handlers.items():
        handler.ingest(rows)
        model = models[name]

        t = start(target_func=lambda h=handler, m=model: inference_loop(h, m, db_util))
        threads.append(t)

    for t in threads:
        t.join()


# --------------------------------------------------
# Stats pipeline (no DataHandler needed)
# --------------------------------------------------


def run_stats_pipeline(db_util: DBUtils) -> None:
    """
    1. Pull process_metrics (full history) and baseline from DB
    2. Filter to ALLOWED_METRICS only
    3. Build per-metric config via baseline_stats logic (in-memory, no JSON file)
    4. Run stats_model_2 pattern detectors
    5. Push results to the `patterns` table
    """
    logger.info(" [Stats] Fetching process_metrics from DB...")
    df = db_util.fetch_all_process_metrics()

    if df.empty:
        logger.error(" [Stats] process_metrics is empty — aborting stats pipeline")
        return

    #  Filter BEFORE groupby loop — only allowed metrics proceed
    df = df[df["metric_name"].isin(ALLOWED_METRICS)]

    if df.empty:
        logger.error(
            " [Stats] No rows remain after ALLOWED_METRICS filter — aborting stats pipeline"
        )
        return

    logger.info(
        " [Stats] %d rows fetched across %d days for %d metrics: %s",
        len(df),
        df["timestamp"].dt.date.nunique(),
        df["metric_name"].nunique(),
        df["metric_name"].unique().tolist(),
    )

    logger.info(" [Stats] Fetching baseline from DB...")
    baseline = db_util.fetch_baseline()  # dict: {metric_name: (mean, std)}

    # ── Build metric config in-memory (mirrors baseline_stats.main()) ──────────
    logger.info(" [Stats] Building per-metric config...")
    results = []
    for (station, metric), _ in df.groupby(["station_name", "metric_name"]):
        r = analyse_metric(metric, station, df, baseline)
        if r:
            results.append(r)

    if not results:
        logger.error(
            "❌ [Stats] analyse_metric returned no results — check data volume (need ≥50 rows per metric/station)"
        )
        return

    metric_cfg = build_config(results)
    logger.info("📊 [Stats] Config built for %d metric/station combos", len(results))

    # ── Run pattern detectors ──────────────────────────────────────────────────
    logger.info("📊 [Stats] Running pattern detectors...")
    patterns_df = run_pattern_pipeline(
        df, global_baseline=baseline, metric_cfg=metric_cfg
    )
    patterns_df = print_summary(patterns_df)

    # ── Push to DB ─────────────────────────────────────────────────────────────
    logger.info("📊 [Stats] Writing patterns to DB...")
    db_util.insert_patterns(patterns_df)

    logger.info(
        "✅ [Stats] Pipeline complete — %d pattern windows written", len(patterns_df)
    )


def main():
    parser = argparse.ArgumentParser(description="Glue Dispenser ML Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "backup", "stats"],
        default="infer",
        help=(
            "Execution mode: train a model, run live inference, "
            "replay from archive logs, or run the stats pattern pipeline"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/analysis_config.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter to a specific model key (e.g. 'glue_flow:xgboost'). Runs all if omitted.",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Path to load a pretrained model from (skips training, loads weights directly)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2026-02-09 00:40:00",
        help="Start timestamp in IST for train/backup mode (format: 'YYYY-MM-DD HH:MM:SS')",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-03-31 19:27:35",
        help="End timestamp in IST for train/backup mode (format: 'YYYY-MM-DD HH:MM:SS')",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=1,
        help="[Infer mode] How often (in seconds) the DB poller checks for new rows",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="Glue_Dispenser",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # Config + DB setup
    # --------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("[DEBUG] Loaded config: %s", cfg)

    DATABASE_URL = cfg["global"]["db"]["database_url_env"]
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set in config")

    # mlflow.set_tracking_uri(args.mlflow_uri)
    # mlflow.set_experiment(args.experiment)

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            inspector = inspect(engine)
            tables = inspector.get_table_names(schema=cfg["global"]["db"].get("schema"))
            if tables:
                logger.info(" Tables found: %s", tables)
            else:
                logger.warning("⚠️ No tables found in schema")
    except Exception:
        logger.exception("Database connection failed")
        raise

    def session_factory():
        return SessionLocal()

    db_util = DBUtils(session_factory=session_factory)

    # --------------------------------------------------
    # Stats mode — no DataHandler or Model needed
    # --------------------------------------------------
    if args.mode == "stats":
        run_stats_pipeline(db_util)
        return

    # --------------------------------------------------
    # Init handlers + models (train / infer / backup only)
    # --------------------------------------------------
    data_handlers = {}
    models = {}

    for target in cfg["target"]:
        for method_config in cfg["target"][target]:
            key = f"{target}:{method_config['method']}"

            if (
                args.model
                and key != args.model
                and method_config["method"] != args.model
            ):
                continue

            handler = DataHandler(config=method_config, target_name=target)
            model = Model(
                data_handler=handler,
                model=method_config["method"],
                config=method_config,
                target_name=target,
            )

            data_handlers[key] = handler
            models[key] = model

    if not models:
        raise RuntimeError(f"No models matched. Check --model value: '{args.model}'")

    # --------------------------------------------------
    # Mode dispatch
    # --------------------------------------------------
    if args.mode == "train":
        start_ts = (
            datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=ZoneInfo("Asia/Kolkata"))
            .astimezone(ZoneInfo("UTC"))
        )
        end_ts = (
            datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=ZoneInfo("Asia/Kolkata"))
            .astimezone(ZoneInfo("UTC"))
        )

        rows = db_util.fetch_data(start_ts, end_ts)

        for name, handler in data_handlers.items():
            handler.ingest(rows)
            XY = handler.fetch_train_data()
            if XY is not None:
                X, y, timestamps = XY
                if X is None:
                    logger.warning(
                        "[%s] fetch_train_data returned no windows — skipping", name
                    )
                    continue
                else:
                    models[name].train(X, y)

    elif args.mode == "backup":
        start_ts = (
            datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=ZoneInfo("Asia/Kolkata"))
            .astimezone(ZoneInfo("UTC"))
        )
        end_ts = (
            datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=ZoneInfo("Asia/Kolkata"))
            .astimezone(ZoneInfo("UTC"))
        )

        infer_from_archive(start_ts, end_ts, data_handlers, models, db_util)

    elif args.mode == "infer":
        poller = DBPoller(
            session_factory=session_factory, poll_interval=args.poll_interval
        )
        poller.start(target_func=lambda: loop(poller, data_handlers, models, db_util))
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
