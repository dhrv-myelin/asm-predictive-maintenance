"""
src/main.py
The Entry Point for the Industrial Digital Twin Engine.
NOW WITH DIRECT TIMESCALEDB STREAMING (NO CSV)
"""
import argparse
from html import parser
import json
import yaml
import time
import os
import sys
import statistics
from datetime import datetime
from collections import defaultdict
from sqlalchemy import text
from datetime import datetime, timedelta
from src.inventory import SystemInventory
from src.log_parser import LogParser
from src.engine import LogicEngine
from src.viz_adapter import VizAdapter
from src.utils.tag_resolver import TagResolver
from src.database import engine as db_engine
import statistics
# Import our modules
#from inventory import SystemInventory
#from log_parser import LogParser
#from engine import LogicEngine
#from viz_adapter import VizAdapter
#from utils.tag_resolver import TagResolver
#from db_manager import TimescaleDBManager
#from src.inventory import SystemInventory
#from src.db_manager import TimescaleDBManager
from src.engine import LogicEngine
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)
    


def run_record_mode(baseline_file, baseline_hours):
    """
    Generate baseline from TimescaleDB process_metrics table.
    """

    print("\n--- RECORD MODE: Generating Baseline ---")
    from src.db.models import BaselineMetric
    from src.database import SessionLocal

    if db_engine is None:
        print("Database engine not initialized.")
        return

    baseline = {}

    with db_engine.connect() as conn:

        # Get earliest timestamp
        start_ts = conn.execute(
            text("SELECT MIN(timestamp) FROM process_metrics")
        ).scalar()

        if start_ts is None:
            print("No metric data found in database.")
            return

        # Apply baseline hour window
        if baseline_hours:
            print(f"Using first {baseline_hours} hours for baseline")
            rows = conn.execute(
                text("""
                    SELECT station_name, metric_name, value
                    FROM process_metrics
                    WHERE timestamp <= :end_time
                """),
                {"end_time": start_ts + timedelta(hours=baseline_hours)}
            ).fetchall()
        else:
            rows = conn.execute(
                text("""
                    SELECT station_name, metric_name, value
                    FROM process_metrics
                """)
            ).fetchall()

    if not rows:
        print("No data retrieved for baseline.")
        return

    print(f"✓ Retrieved {len(rows)} records")

    grouped_data = defaultdict(list)

    for station_name, metric_name, value in rows:
        if value is not None:
            grouped_data[(station_name, metric_name)].append(value)

    if not grouped_data:
        print("No numeric metrics found.")
        return

    print(
        f"\n{'Station':<20} | {'Metric':<30} | "
        f"{'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8} | {'Samples':<7}"
    )
    print("-" * 110)

    for (station, metric), values in grouped_data.items():

        if len(values) < 2:
            continue

        avg = statistics.mean(values)
        std = statistics.stdev(values)
        min_val = min(values)
        max_val = max(values)

        if station not in baseline:
            baseline[station] = {}

        baseline[station][metric] = {
            "mean": round(avg, 6),
            "std_dev": round(std, 6),
            "min_limit": round(min_val, 6),
            "max_limit": round(max_val, 6),
            "sample_count": len(values),
            "created_at": datetime.now().isoformat()
        }

        print(
            f"{station:<20} | {metric:<30} | "
            f"{avg:<8.2f} | {std:<8.4f} | "
            f"{min_val:<8.2f} | {max_val:<8.2f} | {len(values):<7}"
        )

    if not baseline:
        print("Not enough samples to create baseline.")
        return

    os.makedirs(os.path.dirname(baseline_file), exist_ok=True)

    with open(baseline_file, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\n✓ Baseline saved to: {baseline_file}")
    print(f"  Stations: {len(baseline)}")
    print(f"  Total metrics: {sum(len(v) for v in baseline.values())}")
    # --- Save baseline to database ---
    from sqlalchemy.dialects.postgresql import insert

    print("\n--- Saving baseline to database ---")

    with SessionLocal() as session:
        for station, metrics in baseline.items():
            for metric, stats in metrics.items():

                stmt = insert(BaselineMetric).values(
                station_name=station,
                metric_name=metric,
                mean=stats["mean"],
                std_dev=stats["std_dev"],
                min_limit=stats["min_limit"],
                max_limit=stats["max_limit"],
                sample_count=stats["sample_count"],
                unit="seconds"
            )

                stmt = stmt.on_conflict_do_update(
                    index_elements=["station_name", "metric_name"],
                    set_={
                    "mean": stats["mean"],
                    "std_dev": stats["std_dev"],
                    "min_limit": stats["min_limit"],
                    "max_limit": stats["max_limit"],
                    "sample_count": stats["sample_count"],
                    "unit": "seconds"
                }
            )

                session.execute(stmt)

        session.commit()

    print("✓ Baseline upserted into database")


def run_monitor_mode(engine, baseline_file):
    """Loads baseline for future alerting logic"""
    print(f"\n--- MONITOR MODE: Using Baseline {baseline_file} ---")
    if not os.path.exists(baseline_file):
        print("WARNING: Baseline file not found! Running in Data Collection only.")
        print(f"  Generate a baseline first by running: --mode record")
    else:
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
            print(f"✓ Baseline loaded with {len(baseline)} components")
            print("  (Real-time alerting logic goes here in Phase 4)")

def main():
    parser = argparse.ArgumentParser(description="Industrial Log Analytics Engine")
    parser.add_argument('--mode', choices=['record', 'monitor'], required=True)
    parser.add_argument('--input', default='data/machine_logs.txt')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--patterns', default='config/log_patterns/prod_patterns.yaml')
    parser.add_argument('--no-db', action='store_true', help='Disable database (fallback to CSV)')
    parser.add_argument(
    '--baseline_hours',
    type=int,
    default=4,
    help='Hours of data to use for baseline (record mode)'
)
    args = parser.parse_args()
    BASELINE_HOURS = args.baseline_hours

    # 1. Load Configuration
    print("Loading Configs...")
    try:
        graph_config = load_yaml("config/machine_graph.yaml")
        logic_config = load_yaml("config/process_logic.yaml")
        
        print("Loading Hardware Maps...")
        io_config = load_json("config/IOConfig.json")
        gantry_config = load_json("config/GantryConfig.json")
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Missing config file. {e}")
        sys.exit(1)

    # 2. Build System Inventory
    print("Building Inventory...")
    try:
        inventory = SystemInventory.build(graph_config, logic_config)
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)


    # 4. Initialize Visualizer
    start_time_str = None
    try:
        with open(args.input, 'r') as f:
            first_line = f.readline()
            start_time_str = first_line.split(' [')[0]
    except:
        pass

    viz = None
    if args.viz:
        print(f"Initializing Visualizer with Base Time: {start_time_str}")   
        viz = VizAdapter("config/viz_resources.yaml", start_time_str=start_time_str)

    # 5. Initialize Engine
    resolver = TagResolver("config/machine_graph.yaml")

    engine = LogicEngine(
        inventory=inventory, 
        visualizer=viz,
        tag_resolver=resolver,
    )

    # 6. Initialize Parser
    print(f"Loading Patterns from: {args.patterns}")
    if not os.path.exists(args.patterns):
        print(f"CRITICAL: Pattern file not found at {args.patterns}")
        sys.exit(1)

    log_parser = LogParser(args.patterns)

    # 7. Run the Loop
    print(f"\n=== Starting Engine ===")
    print(f"Input: {args.input}")
    print(f"Mode: {args.mode.upper()}")
    print("Storage: TimescaleDB")
    if args.live:
        print(">> LIVE MODE ACTIVE (Press Ctrl+C to stop)")
    print("=" * 50 + "\n")

    line_count = 0
    start_time = time.time()
    last_log_timestamp = None  # Track the last log timestamp as datetime object
    
    try:
        for timestamp, event in log_parser.parse_file(args.input, live_mode=args.live):
            # Track the last timestamp from the log file
            # Convert Unix timestamp to datetime object if needed
            print("PARSED EVENT:", event)
            if isinstance(timestamp, (int, float)):
                last_log_timestamp = datetime.fromtimestamp(timestamp)
            else:
                last_log_timestamp = timestamp
            
            # Normal engine processing
            engine.process_event(timestamp, event)
            
            # ERROR AND WARN LOGGING
            
            
            line_count += 1
            if line_count % 5000 == 0:
                elapsed = time.time() - start_time
                rate = line_count / elapsed
                print(f"Processed {line_count} events... ({int(rate)} ev/s)")

    except KeyboardInterrupt:
        print("\n\n=== Stopping by User Request ===")

    finally:
        # Graceful Shutdown
        print("\n=== Shutting Down ===")
        
        print(f"Engine Stopped. Total Events: {line_count}")
        
        # Generate baseline from database (ONLY in record mode)
        if args.mode == 'record':
            if last_log_timestamp:
                print(f"\nLast log timestamp from file: {last_log_timestamp}")
            else:
                print("\n⚠ Warning: No log timestamp captured, will use current time")
            run_record_mode(
    "data/process_baseline.json",
    args.baseline_hours)
if __name__ == "__main__":
    main()