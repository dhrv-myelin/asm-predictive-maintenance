"""
main.py
--------
Entrypoint for Grafana dashboard generation.

Usage:
    python main.py                                  # uses default config
    python main.py --config config/my_station.yaml  # specify a config file
    python main.py --output output/my_dashboard.json

To generate dashboards for multiple stations, run once per config file.
"""

import argparse
import json
import os
import sys

from builder.dashboard_builder import build_dashboard


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grafana dashboard JSON from config.")
    parser.add_argument(
        "--config",
        default="config/ced_maintenance.yaml",
        help="Path to the YAML config file (default: config/ced_maintenance.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the output JSON file (default: output/<dashboard_uid>.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    print(f"[INFO] Loading config: {args.config}")
    dashboard = build_dashboard(args.config)

    # Determine output path
    output_path = args.output
    if output_path is None:
        os.makedirs("output", exist_ok=True)
        uid = dashboard.get("uid", "dashboard")
        output_path = f"output/{uid}.json"

    # Write output
    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    panel_count = len(dashboard.get("panels", []))
    print(f"[OK] Dashboard '{dashboard['title']}' generated with {panel_count} panels.")
    print(f"[OK] Output written to: {output_path}")


if __name__ == "__main__":
    main()
