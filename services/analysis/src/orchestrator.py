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
                        model_name="mamba",  # model.model_name
                    )

                    # move window
                    curr_ts_map[name] = window.iloc[0]["timestamp"]

        except Exception as e:
            logger.exception("Orchestrator error: %s", e)

        time.sleep(poller.poll_interval)


def inference_loop(data_handler, model, db_util):
    curr_first_timestamp = None
    while True:
        X = data_handler.fetch_next_window(curr_first_timestamp, for_training=False)

        if X is None:
            print("❌ Not enough data for window")
            time.sleep(1)  # prevent CPU spin
            continue

        curr_first_timestamp = X.iloc[0]["timestamp"]

        print("✅ Window shape:", X.shape)
        # print(X)
        # print("[DEBUG] Window:\n", X)
        # inference
        preds = model.real_time_inference(X)
        preds = [preds[-1]]
        # print("================= Length Of Predictions:", len(preds))

        # write results
        last_ts = X.iloc[-1]["timestamp"]
        print("[DEBUG] Last timestamp in window:", last_ts)
        db_util.insert_results(
            last_timestamp=last_ts,
            values=preds,  #  [0.0] * X.shape[0]
            station_name=data_handler.target_name.split("__")[0],
            metric_name=data_handler.target_name.split("__")[1],
            model_name="mamba",  # model.model_name
        )

        time.sleep(0.5)  # pacing


def infer_from_archive(start_ts, end_ts, data_handlers, models, db_util):
    rows = db_util.fetch_data(start_ts, end_ts)

    if not rows:
        # print(f"[ERROR] No data found between {start_ts} and {end_ts}")
        return

    def start(target_func):
        t = threading.Thread(target=target_func)
        t.start()

    threads = []
    for name, handler in data_handlers.items():
        handler.ingest(rows)
        model = models[name]

        t = start(target_func=lambda h=handler, m=model: inference_loop(h, m, db_util))
        threads.append(t)

    # block forever (or join threads)
    for t in threads:
        t.join()


# ====================== new main =============================


def main():
    parser = argparse.ArgumentParser(description="Glue Dispenser ML Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "backup"],
        default="infer",
        help="Execution mode: train a model, run live inference, or replay from archive logs",
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
        default="2026-01-23 22:06:00",
        help="Start timestamp for train/backup mode (format: 'YYYY-MM-DD HH:MM:SS')",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-01-23 23:10:35",
        help="End timestamp for train/backup mode (format: 'YYYY-MM-DD HH:MM:SS')",
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

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful")
            inspector = inspect(engine)
            tables = inspector.get_table_names(schema=cfg["global"]["db"].get("schema"))
            if tables:
                logger.info("📦 Tables found: %s", tables)
            else:
                logger.warning("⚠️ No tables found in schema")
    except Exception:
        logger.exception("❌ Database connection failed")
        raise

    def session_factory():
        return SessionLocal()

    db_util = DBUtils(session_factory=session_factory)

    # --------------------------------------------------
    # Init handlers + models
    # --------------------------------------------------
    data_handlers = {}
    models = {}

    for target in cfg["target"]:
        for method_config in cfg["target"][target]:
            key = f"{target}:{method_config['method']}"

            # If --model is specified, skip everything else
            if args.model and key != args.model:
                continue

            handler = DataHandler(config=method_config, target_name=target)
            model = Model(
                data_handler=handler,
                model=method_config["method"],
                config=method_config,
                target_name=target,
            )

            if args.load_path:
                logger.info("📂 Loading model weights from: %s", args.load_path)
                model.load(
                    args.load_path
                )  # assumes your Model class has a load() method

            data_handlers[key] = handler
            models[key] = model

    if not models:
        raise RuntimeError(f"No models matched. Check --model value: '{args.model}'")

    # --------------------------------------------------
    # Mode dispatch
    # --------------------------------------------------
    if args.mode == "train":
        start_ts = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
        end_ts = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
        rows = db_util.fetch_data(start_ts, end_ts)

        for name, handler in data_handlers.items():
            handler.ingest(rows)
            XY = handler.fetch_train_data()
            if XY:
                X, y = XY
                logger.info("[%s] X shape: %s | y shape: %s", name, X.shape, y.shape)
                models[name].train(X, y)

    elif args.mode == "backup":
        start_ts = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
        end_ts = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
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
# # --------------------------------------------------
# # Bootcolumns
# # --------------------------------------------------
#
# if __name__ == "__main__":

# TRAIN = True
# BACKUP_LOGS = False
# start_ts = datetime(2026, 1, 23, 22, 6, 0)
# end_ts = datetime(2026, 1, 23, 23, 10, 35)
#
# config_path = "config/analysis_config.yaml"
# with open(config_path, "r") as f:
#     cfg = yaml.safe_load(f)
# print("[DEBUG] Loaded config : ", cfg)
# # resolve DB URL
# # DATABASE_URL = os.getenv(cfg['global']['db']['database_url_env'])
# # INFO: removed getenv bc im not making it a env
# DATABASE_URL = cfg["global"]["db"]["database_url_env"]
# if not DATABASE_URL:
#     raise RuntimeError("DATABASE_URL env var not set")
#
# # mlflow uri
# import os
# import mlflow
#
# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("Glue_Dispenser")
#
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(bind=engine)
# try:
#     with engine.connect() as conn:
#         conn.execute(text("SELECT 1"))
#         logger.info("✅ Database connection successful")
#
#         inspector = inspect(engine)
#         tables = inspector.get_table_names(
#             schema=cfg["global"]["db"].get("schema", None)
#         )
#
#         if tables:
#             logger.info("📦 Tables found in DB:")
#             for t in tables:
#                 logger.info(f"   - {t}")
#         else:
#             logger.warning("⚠️ No tables found in database schema")
#
# except Exception as e:
#     logger.exception("❌ Database connection failed")
#     raise
#
# def session_factory():
#     return SessionLocal()
#
# db_util = DBUtils(session_factory=session_factory)
#
# # --------------------------------------------------
# # Init handlers, models
# # --------------------------------------------------
#
# data_handlers = {}
# models = {}
#
# for target in cfg["target"]:
#     for method_config in cfg["target"][target]:
#
#         handler = DataHandler(config=method_config, target_name=target)
#
#         model = Model(
#             data_handler=handler,
#             model=method_config["method"],
#             config=method_config,
#             target_name=target,
#         )
#
#         key = f"{target}:{method_config['method']}"
#
#         data_handlers[key] = handler
#         models[key] = model
#
# # --------------------------------------------------
# # Training mode
# # --------------------------------------------------
#
# if TRAIN:
#     rows = db_util.fetch_data(start_ts, end_ts)
#     for name, handler in data_handlers.items():
#         handler.ingest(rows)
#         model = models[name]
#         XY = handler.fetch_train_data()
#         print("[DEBUG] Train data : \n", XY)
#         if XY:
#             X, y = XY
#             print("[DEBUG] Train X shape:", X.shape)
#             print("[DEBUG] Train y shape:", y.shape)
#             models[name].train(X, y)
#
# elif BACKUP_LOGS:
#     infer_from_archive(start_ts, end_ts, data_handlers, models, db_util)
# # --------------------------------------------------
# # Inference mode
# # --------------------------------------------------
#
# else:
#     poller = DBPoller(session_factory=session_factory, poll_interval=1)
#     poller.start(target_func=lambda: loop(poller, data_handlers, models, db_util))
#
#     while True:
#         time.sleep(60)
