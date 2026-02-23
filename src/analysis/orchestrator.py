from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from data_handler import DataHandler
from poller import DBPoller
from model import Model
from db_utils import DBUtils
import yaml
import time
import os
import logging
import threading
from datetime import datetime

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
            # print("[DEBUG] Polling for data...")
            rows = poller.poll()
            # print("[DEBUG] Rows : ",rows)
            if not rows:
                time.sleep(poller.poll_interval)
                continue

            for name, handler in data_handlers.items():

                handler.ingest(rows)

                window = handler.fetch_next_window(
                    curr_first_timestamp=curr_ts_map[name],
                    for_training=False
                )

                if window is not None:
                    inference_counter += 1
                    print(f"[DEBUG] Inference number : {inference_counter}")
                    model = models[name]

                    # inference
                    preds = model.real_time_inference(window)

                    # write
                    last_ts = window.iloc[-1]["timestamp"]
                    db_util.insert_results(last_ts, preds)

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
            time.sleep(1)   # prevent CPU spin
            continue

        curr_first_timestamp = X.iloc[0]['timestamp']

        print("✅ Window shape:", X.shape)
        # print(X)

        # inference
        # preds = model.real_time_inference(X)

        # write results
        last_ts = X.iloc[-1]['timestamp']
        db_util.insert_results(
            last_timestamp=last_ts,
            values= [0.0] * X.shape[0], # preds
            station_name = data_handler.target_name.split("__")[0],
            metric_name = data_handler.target_name.split("__")[1],
            model_name = model.name
        )

        time.sleep(0.5)   # pacing


def infer_from_archive(start_ts, end_ts, data_handlers, models, db_util):
    rows = db_util.fetch_data(start_ts, end_ts)

    if not rows:
        print(f"[ERROR] No data found between {start_ts} and {end_ts}")
        return

    def start(target_func):
        t = threading.Thread(target=target_func)
        t.start()
    threads = []
    for name, handler in data_handlers.items():
        handler.ingest(rows)
        model = models[name]

        t = start(
            target_func=lambda h=handler, m=model: inference_loop(h, m, db_util)
        )
        threads.append(t)

    # block forever (or join threads)
    for t in threads:
        t.join()

# --------------------------------------------------
# Bootcolumns
# --------------------------------------------------

if __name__ == "__main__":

    TRAIN = False
    BACKUP_LOGS = True
    start_ts = datetime(2026, 1, 24, 3, 36, 0)
    end_ts   = datetime(2026, 1, 24, 4, 42, 0)

    config_path = "../../config/analysis_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # resolve DB URL
    DATABASE_URL = os.getenv(cfg['global']['db']['database_url_env'])
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL env var not set")

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful")

            inspector = inspect(engine)
            tables = inspector.get_table_names(schema=cfg['global']['db'].get('schema', None))

            if tables:
                logger.info("📦 Tables found in DB:")
                for t in tables:
                    logger.info(f"   - {t}")
            else:
                logger.warning("⚠️ No tables found in database schema")

    except Exception as e:
        logger.exception("❌ Database connection failed")
        raise

    def session_factory():
        return SessionLocal()
    
    db_util = DBUtils(
        session_factory=session_factory
    )

    # --------------------------------------------------
    # Init handlers, models
    # --------------------------------------------------

    data_handlers = {}
    models = {}

    for target in cfg['target']:
        for method_config in cfg['target'][target]:

            handler = DataHandler(
                config=method_config,
                target_name=target
            )

            model = Model(
                data_handler=handler,
                model=method_config['method'],
                config=method_config,
                target_name=target
            )

            key = f"{target}:{method_config['method']}"

            data_handlers[key] = handler
            models[key] = model

    # --------------------------------------------------
    # Training mode
    # --------------------------------------------------

    if TRAIN:
        rows = db_util.fetch_data(start_ts, end_ts)
        for name, handler in data_handlers.items():
            handler.ingest(rows)
            model = models[name]
            XY = handler.fetch_train_data()
            print("[DEBUG] Train data : \n", XY)
            if XY:
                X, y = XY
                # models[name].train(X, y)
    
    elif BACKUP_LOGS:
        infer_from_archive(start_ts, end_ts, data_handlers, models, db_util)
    # --------------------------------------------------
    # Inference mode
    # --------------------------------------------------

    else:
        poller = DBPoller(
            session_factory=session_factory,
            poll_interval=1
        )
        poller.start(
            target_func=lambda: loop(poller, data_handlers, models, db_util)
        )

        while True:
            time.sleep(60)
