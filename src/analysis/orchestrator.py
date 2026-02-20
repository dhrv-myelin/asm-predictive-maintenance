from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data_handler import DataHandler
from poller import DBPoller
from model import Model
import yaml
import time

import logging

logger = logging.getLogger(__name__)

def loop(poller):
    while not poller.if_stop.is_set():
        try:
            rows = poller.poll()
            if rows:
                for dl in data_handlers: # get all the data handlers here
                    window = dl.ingest(rows)
                    if window:
                        # During initialisation, Data handlerrs and Model Classes are initialiesd together based on the config
                        # Identify which model to send the data to form that some how
                        inf_results = model.real_time_inference()
                        db.insert_results(inf_results)

        except Exception as e:
            logger.exception("Poller error: %s", e)
        time.sleep(poller.poll_interval)

    
if __name__ == "__main__":

    TRAIN = True

    config_path="../../config/analysis_config.yaml"
    with open(config_path, "r") as f: 
        cfg = yaml.safe_load(f) 

    DATABASE_URL = cfg['global']['db']['database_url_env']

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)

    def session_factory():
        return SessionLocal()
    
    # => Initialise DB from db.py Connection here

    # Initialising DataHandlers and ModelClasses
    for target in cfg['target']:
        for method_config in cfg['target'][target]:
            handler = DataHandler(
                session_factory=session_factory,
                config=method_config,
                target_name=target
            )
            model = Model(method_config, method_config['method'], target)

    if TRAIN:
        rows = db.fetch_train_data(start_timestamp, end_timestamp)
        train_data, val_data = handler.format_train
        #corrresponding model to train 
        model.train(train_data, val_data)
        
    else: 
        poller = DBPoller(
            session_factory=session_factory,
            poll_interval=1
            )
        poller.start(target_func=loop)
