"""
src/main.py
The Entry Point for the Industrial Digital Twin Engine.

Orchestrates the Inventory, Parser, Logic Engine, and Visualizer.

Arguments:
  --mode      [Required] 'record' (generate baseline) or 'monitor' (detect anomalies)
  --input     [Optional] Path to log file (default: data/machine_logs.txt)
  --patterns  [Optional] Path to regex config (default: config/patterns/prod_patterns.yaml)
  --viz       [Flag]     Enable Rerun.io 3D Visualization
  --live      [Flag]     Tail the log file indefinitely (Real-time mode)

Usage Examples:
  # 1. Sanity Check (Synthetic Logs + Test Patterns + Visualizer)
  python src/main.py --mode record --viz --patterns config/log_patterns/test_patterns.yaml

  # 2. Production Baseline (Real Logs + Production Patterns)
  python src/main.py --mode record --input data/real_logs.txt --patterns config/log_patterns/prod_patterns.yaml

  # 3. Production Monitoring (Live Tailing)
  python src/main.py --mode monitor --live --input data/real_logs.txt
"""
import argparse
import json
import yaml
import time
import os
import sys
import csv
import statistics
from datetime import datetime
from collections import defaultdict

# Import our modules
from inventory import SystemInventory
from log_parser import LogParser
from engine import LogicEngine
from viz_adapter import VizAdapter
from utils.tag_resolver import TagResolver

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def load_json(path):
    if not os.path.exists(path):
        return {} # Return empty dict if optional config is missing
    with open(path, 'r') as f:
        return json.load(f)

def run_record_mode(engine, baseline_file):
    """
    Generates Baseline using standard Python libraries only.
    No Pandas required.
    """
    
    print("\n--- RECORD MODE: Generating Baseline ---")
    
    # 1. Flush Buffer
    engine.flush_buffer()
    
    # 2. Read CSV and Aggregate Data
    # Structure: dict of lists -> { ('l1_dispenser', 'cycle_time'): [12.0, 12.5, ...] }
    grouped_data = defaultdict(list)
    
    try:
        with open(engine.output_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None) # Skip header
            if not header:
                print("No data recorded.")
                return

            for row in reader:
                # CSV: [timestamp, comp_id, metric_name, value, unit, context...]
                if len(row) < 4: continue
                
                comp_id = row[1]
                metric = row[2]
                try:
                    val = float(row[3])
                    grouped_data[(comp_id, metric)].append(val)
                except ValueError:
                    continue

    except FileNotFoundError:
        print("No CSV file found.")
        return

    # 3. Calculate Statistics
    baseline = {}
    
    print(f"{'Component':<20} | {'Metric':<20} | {'Mean':<8} | {'StdDev':<8}")
    print("-" * 65)

    for (comp_id, metric), values in grouped_data.items():
        if len(values) < 2:
            print(f"Skipping {comp_id}.{metric} (Not enough data points)")
            continue
            
        avg = statistics.mean(values)
        std = statistics.stdev(values)
        min_v = min(values)
        max_v = max(values)
        
        # Initialize dict structure
        if comp_id not in baseline: baseline[comp_id] = {}
        
        # Logic: Normal = Mean +/- 3 Sigma (or 10% min buffer)
        # If variance is tiny (std ~ 0), we force a small tolerance so alerts don't spam.
        safe_std = std if std > 0 else (avg * 0.05)
        
        baseline[comp_id][metric] = {
            "mean": round(avg, 2),
            "std_dev": round(std, 4),
            "min_limit": round(min_v * 0.9, 2), # 10% below historical min
            "max_limit": round(max_v * 1.1, 2)  # 10% above historical max
        }
        
        print(f"{comp_id:<20} | {metric:<20} | {avg:<8.2f} | {std:<8.2f}")

    # 4. Save to JSON
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, indent=2)
        
    print(f"\nSuccess! Baseline saved to: {baseline_file}")

def run_monitor_mode(engine, baseline_file):
    """
    Loads baseline and (in future) injects it into Engine for real-time alerting.
    For MVP, the Engine just records data. We can add a 'Check vs Baseline' hook here.
    """
    print(f"\n--- MONITOR MODE: Using Baseline {baseline_file} ---")
    if not os.path.exists(baseline_file):
        print("WARNING: Baseline file not found! Running in Data Collection only.")
    else:
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
            print("Baseline loaded. (Real-time alerting logic goes here in Phase 4)")

def main():
    parser = argparse.ArgumentParser(description="Industrial Log Analytics Engine")
    parser.add_argument('--mode', choices=['record', 'monitor'], required=True, help="Operation Mode")
    parser.add_argument('--input', default='data/machine_logs.txt', help="Path to log file")
    parser.add_argument('--live', action='store_true', help="Tail the file (Infinite Loop)")
    parser.add_argument('--viz', action='store_true', help="Enable Visualization (Rerun)")
    parser.add_argument('--patterns', default='config/log_patterns/prod_patterns.yaml', help="Path to regex config")
    args = parser.parse_args()

    # 1. Load Configuration
    print("Loading Configs...")
    try:
        # 1. Essential Configs
        graph_config = load_yaml("config/machine_graph.yaml")
        logic_config = load_yaml("config/process_logic.yaml")

        # 2. Hardware Configs (Future-Proofing the Inputs)
        # These are not currently used by Regex Parser, but loaded for the Engine's Inventory
        print("Loading Hardware Maps...")
        io_config = load_json("config/IOConfig.json")
        gantry_config = load_json("config/GantryConfig.json")
        # log_patterns is loaded by Parser internally
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Missing config file. {e}")
        sys.exit(1)

    # 2. Build System (Inventory)
    print("Building Inventory...")
    try:
        inventory = SystemInventory.build(graph_config, logic_config)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    # 3. Initialize Visualizer if requested
    start_time_str = None
    try:
        with open(args.input, 'r') as f:
            first_line = f.readline()
            # Assuming format: "2026-01-21 10:00:00,000 ..."
            start_time_str = first_line.split(' [')[0] # Grab date part
    except:
        pass

    viz = None
    if args.viz:
        print(f"Initializing Visualizer with Base Time: {start_time_str}")   
        viz = VizAdapter("config/viz_resources.yaml", start_time_str=start_time_str)


    # 4. Initialize Engine

    resolver = TagResolver("config/machine_graph.yaml")

    engine = LogicEngine(
        inventory=inventory, 
        output_file="data/process_metrics.csv",
        visualizer=viz,
        tag_resolver=resolver
    )

    # 5. Initialize Parser
    print(f"Loading Patterns from: {args.patterns}")
    if not os.path.exists(args.patterns):
        print(f"CRITICAL: Pattern file not found at {args.patterns}")
        sys.exit(1)

    log_parser = LogParser(args.patterns)

    # 6. Run the Loop
    print(f"Starting Engine on: {args.input}")
    if args.live:
        print(">> LIVE MODE ACTIVE (Press Ctrl+C to stop)")

    line_count = 0
    start_time = time.time()

    try:
        # The parser generator handles the "Live vs Static" logic internally now
        # We just pass the flag

        for timestamp, event in log_parser.parse_file(args.input, live_mode=args.live):
            engine.process_event(timestamp, event)
            line_count += 1
            
            # Progress Heartbeat (every 5000 lines)
            if line_count % 5000 == 0:
                elapsed = time.time() - start_time
                rate = line_count / elapsed
                print(f"Processed {line_count} events... ({int(rate)} ev/s)")

    except KeyboardInterrupt:
        print("\nStopping by User Request...")

    finally:
        # Graceful Shutdown
        engine.flush_buffer()
        print(f"Engine Stopped. Total Events: {line_count}")
        
        # If in Record Mode, generate the JSON now
        if args.mode == 'record':
            run_record_mode(engine, "data/process_baseline.json")

if __name__ == "__main__":
    main()
