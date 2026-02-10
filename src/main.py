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

# Import our modules
from inventory import SystemInventory
from log_parser import LogParser
from engine import LogicEngine
from viz_adapter import VizAdapter
from utils.tag_resolver import TagResolver
from db_manager import TimescaleDBManager

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)
    


def run_record_mode(engine, db_manager, baseline_file,baseline_hours,last_log_timestamp=None):
    """
    Generates Baseline from TimescaleDB
    
    Args:
        engine: LogicEngine instance
        db_manager: TimescaleDBManager instance
        baseline_file: Path to save baseline JSON
        last_log_timestamp: Timestamp of the last log line processed (datetime)
    """
    print("\n--- RECORD MODE: Generating Baseline ---")
    
    if not db_manager or not db_manager.enabled:
        print("ERROR: Database not available. Cannot generate baseline.")
        print("Tip: Check config/db_config.json and ensure TimescaleDB is running")
        return

    # Load existing baseline if present
    if os.path.exists(baseline_file):
        with open(baseline_file, "r") as f:
            baseline = json.load(f)
        print(f"✓ Loaded existing baseline: {baseline_file}")
    else:
        baseline = {}
        print("✓ No existing baseline found. Creating new one.")

    # Fetch all metric data from database
    print("Reading metrics from database...")
    rows = db_manager.get_metrics_for_baseline()

    # Get earliest timestamp
    with db_manager.engine.connect() as conn:
        start_ts = conn.execute(text("""
            SELECT MIN(timestamp) FROM process_metrics
        """)).scalar()

        if start_ts is None:
            print("No data found.")
            return

        if baseline_hours is not None:
            print(f"Using first {baseline_hours} hours for baseline")

            rows = conn.execute(text("""
                SELECT station_name, metric_name, value
                FROM process_metrics
                WHERE timestamp <= :end_time
            """), {
                "end_time": start_ts + timedelta(hours=baseline_hours)
            }).fetchall()
        else:
            rows = conn.execute(text("""
                SELECT station_name, metric_name, value
                FROM process_metrics
            """)).fetchall()

    print(f"✓ Retrieved {len(rows)} records")


    if not rows:
        print("No metric data found. Baseline not updated.")
        return

    print(f"✓ Found {len(rows)} metric records")

    # Group values by (component, metric)
    grouped_data = defaultdict(list)

    for comp_id, metric, value in rows:
        if value is not None:
            grouped_data[(comp_id, metric)].append(value)

    if not grouped_data:
        print("No valid numeric metrics found.")
        return

    print(
        f"\n{'Component':<20} | {'Metric':<30} | "
        f"{'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8} | {'Samples':<7}"
    )
    print("-" * 110)

    # Compute statistics
    for (comp_id, metric), values in grouped_data.items():

        if len(values) < 2:
            print(f"Skipping {comp_id}.{metric} (only {len(values)} sample)")
            continue

        avg = statistics.mean(values)
        std = statistics.stdev(values)
        observed_min = min(values)
        observed_max = max(values)

        # Ensure component exists
        if comp_id not in baseline:
            baseline[comp_id] = {}

        # If metric exists → expand limits
        if metric in baseline[comp_id]:
            old_entry = baseline[comp_id][metric]

            old_min = old_entry.get("min_limit", observed_min)
            old_max = old_entry.get("max_limit", observed_max)

            baseline[comp_id][metric] = {
                "mean": round(avg, 6),
                "std_dev": round(std, 6),
                "min_limit": round(min(old_min, observed_min), 6),
                "max_limit": round(max(old_max, observed_max), 6),
                "sample_count": len(values),
                "record_time": datetime.now().isoformat()
            }

        # New metric → create fresh baseline
        else:
            baseline[comp_id][metric] = {
                "mean": round(avg, 6),
                "std_dev": round(std, 6),
                "min_limit": round(observed_min, 6),
                "max_limit": round(observed_max, 6),
                "sample_count": len(values),
                "record_time": datetime.now().isoformat()
            }

        print(
            f"{comp_id:<20} | {metric:<30} | "
            f"{avg:<8.2f} | {std:<8.4f} | "
            f"{observed_min:<8.2f} | {observed_max:<8.2f} | {len(values):<7}"
        )
    if not baseline:
        print("\nNo baseline could be generated. Need at least 2 samples per metric.")
        return

    # Save to JSON
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, indent=2)
        
    print(f"\n✓ Success! Baseline saved to: {baseline_file}")
    print(f"  Components: {len(baseline)}")
    print(f"  Total metrics: {sum(len(metrics) for metrics in baseline.values())}")
    
    # ========================================================================
    # Load baseline into database and merge timestamps
    # ========================================================================
    print("\n--- Loading Baseline into Database ---")
    if db_manager.load_baseline_from_json(baseline_file):
        print("\n--- Merging Timestamps ---")
        db_manager.merge_timestamps_to_baseline(last_log_timestamp=last_log_timestamp)
    else:
        print("⚠ Skipping timestamp merge - baseline load failed")
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

    # 3. Initialize Database Manager
    db_manager = None
    if not args.no_db:
        print("\n=== Initializing TimescaleDB ===")
        try:
            db_manager = TimescaleDBManager("config/db_config.json")
            
            if db_manager.enabled:
                print("✓ TimescaleDB streaming enabled")
                # Test the connection
                try:
                    if db_manager.engine:
                        with db_manager.engine.connect() as conn:
                            result = conn.execute(text("SELECT version();"))
                            version_info = result.fetchone()[0][:50]
                            print(f"✓ Database connection verified: {version_info}...")
                    else:
                        print("✗ Engine not initialized.")
                        db_manager.enabled = False
                except Exception as e:
                    print(f"✗ Connection test failed: {e}")
                    db_manager.enabled = False
            else:
                print("✗ Database failed to initialize - check errors above")
                print(f"  Config loaded: {db_manager.config}")
                
        except Exception as e:
            print(f"✗ Fatal error during DB setup: {e}")
            db_manager = None


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
        output_file="data/process_metrics.csv",  # Fallback CSV
        visualizer=viz,
        tag_resolver=resolver,
        db_manager=db_manager
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
    if db_manager and db_manager.enabled:
        print(f"Storage: TimescaleDB ({db_manager.config['database']})")
    else:
        print(f"Storage: CSV (fallback)")
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
            if isinstance(timestamp, (int, float)):
                last_log_timestamp = datetime.fromtimestamp(timestamp)
            else:
                last_log_timestamp = timestamp
            
            # Normal engine processing
            engine.process_event(timestamp, event)
            
            # ERROR AND WARN LOGGING
            if event.get("level") in ["ERROR", "WARN"] and db_manager and db_manager.enabled:
                raw_line = event.get("raw_line", "")
                parts = raw_line.split(' ', 4)
                
                if len(parts) >= 5:
                    log_timestamp = parts[0] + " " + parts[1]
                    severity = parts[3]
                    
                    rest = parts[4].split(' - ', 1)
                    service_name = rest[0].strip() if len(rest) > 0 else ""
                    log_message = rest[1].strip() if len(rest) > 1 else ""
                    
                    station_name = event.get("target", "")
                    thread_id = parts[2].strip('[]') if len(parts) > 2 else ""
                    extra = f"thread_id={thread_id}"
                    
                    # Convert timestamp string to datetime
                    try:
                        dt = datetime.strptime(log_timestamp, "%Y-%m-%d %H:%M:%S,%f")
                    except:
                        dt = datetime.now()
                    
                    db_manager.insert_error(
                        dt, station_name, severity, service_name, log_message, extra
                    )
            
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
            run_record_mode(engine, db_manager, "data/process_baseline.json",args.baseline_hours,last_log_timestamp=last_log_timestamp)
        
        # Close database connection
        if db_manager:
            db_manager.close()

if __name__ == "__main__":
    main()