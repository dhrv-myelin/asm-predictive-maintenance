"""
src/main.py
The Entry Point for the Industrial Digital Twin Engine.
NOW WITH DIRECT TIMESCALEDB STREAMING (NO CSV)
"""
import argparse
import json
import yaml
import time
import os
import sys
import statistics
from datetime import datetime
from collections import defaultdict
from sqlalchemy import text

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

def run_record_mode(engine, db_manager, baseline_file):
    """
    Generates Baseline from TimescaleDB
    """
    print("\n--- RECORD MODE: Generating Baseline ---")
    
    if not db_manager or not db_manager.enabled:
        print("ERROR: Database not available. Cannot generate baseline.")
        print("Tip: Check config/db_config.json and ensure TimescaleDB is running")
        return
    
    # Read data from database
    print("Reading metrics from database...")
    rows = db_manager.get_metrics_for_baseline()
    
    if not rows:
        print("No data recorded in database.")
        print("Possible causes:")
        print("  1. No log events were processed")
        print("  2. Database connection failed during processing")
        print("  3. No metrics were generated from the logs")
        return
    
    print(f"✓ Found {len(rows)} metric records")
    
    # Aggregate data
    grouped_data = defaultdict(list)
    
    for row in rows:
        comp_id = row[0]
        metric = row[1]
        value = row[2]
        
        if value is not None:
            grouped_data[(comp_id, metric)].append(value)
    
    if not grouped_data:
        print("No valid metrics found for baseline calculation.")
        return
    
    # Calculate Statistics
    baseline = {}
    
    print(f"\n{'Component':<20} | {'Metric':<20} | {'Mean':<8} | {'StdDev':<8} | {'Samples':<8}")
    print("-" * 80)

    for (comp_id, metric), values in grouped_data.items():
        if len(values) < 2:
            print(f"Skipping {comp_id}.{metric} (Only {len(values)} data point(s))")
            continue
            
        avg = statistics.mean(values)
        std = statistics.stdev(values)
        min_v = min(values)
        max_v = max(values)
        
        if comp_id not in baseline:
            baseline[comp_id] = {}
        
        safe_std = std if std > 0 else (avg * 0.05)
        
        baseline[comp_id][metric] = {
            "mean": round(avg, 2),
            "std_dev": round(std, 4),
            "min_limit": round(min_v * 0.9, 2),
            "max_limit": round(max_v * 1.1, 2),
            "sample_count": len(values)
        }
        
        print(f"{comp_id:<20} | {metric:<20} | {avg:<8.2f} | {std:<8.4f} | {len(values):<8}")

    if not baseline:
        print("\nNo baseline could be generated. Need at least 2 samples per metric.")
        return

    # Save to JSON
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, indent=2)
        
    print(f"\n✓ Success! Baseline saved to: {baseline_file}")
    print(f"  Components: {len(baseline)}")
    print(f"  Total metrics: {sum(len(metrics) for metrics in baseline.values())}")

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
    args = parser.parse_args()

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
    
    try:
        for timestamp, event in log_parser.parse_file(args.input, live_mode=args.live):
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
            run_record_mode(engine, db_manager, "data/process_baseline.json")
        
        # Close database connection
        if db_manager:
            db_manager.close()

if __name__ == "__main__":
    main()