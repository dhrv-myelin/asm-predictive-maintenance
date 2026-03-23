"""
Excel to YAML Converter - Production Ready
===========================================

Converts GDM_process_details_v2.xlsx to:
- process_logic.yaml
- machine_graph.yaml
- event_log_mapping.yaml
"""

import pandas as pd
import yaml
from collections import defaultdict
from typing import Dict, List, Any, Optional


# for "" in yaml dump
class QuotedStringDumper(yaml.SafeDumper):
    pass


QuotedStringDumper.add_representer(
    str,
    lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", data, style='"'
    ),
)


# INFO: stable
def build_machine_graph(excel_file: str):  # -> Dict[str, Any]
    """
    Build machine_graph.yaml from Excel

    Uses:
    - Station Overview → topology
    - HardwareMap → hardware_link
    """

    # import relevant sheets and add them to pds
    station_df = pd.read_excel(excel_file, sheet_name="Station Overview")
    hardware_df = pd.read_excel(excel_file, sheet_name="HardwareMap")

    # TODO: MAKE TIS VARIABLE
    # graph heading
    machine_graph = {
        "machine": {"name": "Glue_Dispenser_Standardized", "version": "3.0"},
        "topology": [],
    }

    for idx, row in station_df.iterrows():
        station_id = row["Station ID"]

        node = {
            "id": station_id,
            "class": row["Station Class"],
            "logic_template": row["Logic Template"],
        }

        # Add is_entry for first station
        if idx == 0:
            node["is_entry"] = True

        # get state lists
        next_states: str = str(row["Next Nodes (Flow)"])

        # ERROR: dont understand this split error
        next_nodes: List[str] = [s.strip() for s in next_states.split(",")]

        if "END" in next_nodes:
            node["is_exit"] = True

        node["next_nodes"] = next_nodes

        #
        # Build hardware_link from HardwareMap
        station_hw = hardware_df[hardware_df["Station ID"] == station_id]
        hardware_link = defaultdict(dict)

        # INFO: REQUIRE THESE NAMES TO BE STANDARDIZED
        for _, hw_row in station_hw.iterrows():
            category = hw_row["Category"]
            function = hw_row["Function (Role)"]
            hardware_tag = hw_row["Hardware Tag / Variable"]

            # INFO: if none are empty. need to add more such checks
            if pd.notna(category) and pd.notna(function) and pd.notna(hardware_tag):
                category = category.lower()
                hardware_link[category][function] = hardware_tag

            # INFO: MAY NEED TO .lower() other stuff too

        node["hardware_link"] = dict(hardware_link) if hardware_link else {}

        machine_graph["topology"].append(node)

    return machine_graph


def build_process_logic(excel_file: str) -> Dict[str, Any]:
    """
    Build process_logic.yaml from Excel

    Uses:
    - Process State Machine → states and transitions
    - HardwareMap → state_inference
    - Metrics → metrics per state
    """
    station_df = pd.read_excel(excel_file, sheet_name="Station Overview")
    state_machine_df = pd.read_excel(excel_file, sheet_name="Process State Machine")
    hardware_df = pd.read_excel(excel_file, sheet_name="HardwareMap")
    metrics_df = pd.read_excel(excel_file, sheet_name="Metrics")

    process_logic = {
        "version": "1.0",
        "description": "Process Logic",
        "definitions": {},
    }

    # Get unique logic templates from CLEANED station overview
    station_df_clean = station_df[station_df["Logic Template"].notna()]
    logic_templates = station_df_clean["Logic Template"].unique()

    for template in logic_templates:
        # IMPORTANT: Process State Machine sheet uses template names directly as Station ID
        # NOT actual station instances (like l1_buffer_a, l2_buffer_a)
        # So we query by template name itself
        station_states = state_machine_df[state_machine_df["Station ID"] == template]

        if station_states.empty:
            print(f"Warning: No state machine data found for template '{template}'")
            continue

        # Build state machine
        state_machine = build_state_machine(
            template,
            template,  # Pass template name as station_id (that's what's in the sheet)
            station_states,
            metrics_df,
            hardware_df,
        )

        process_logic["definitions"][template] = state_machine

    return process_logic


def build_metric_lookup(metrics_df, template_name):
    """
    Returns:
        {(current_state, event): [metrics]}
    """

    template_metrics = metrics_df[metrics_df["Logic Template"] == template_name]

    metric_lookup = defaultdict(list)

    for _, row in template_metrics.iterrows():
        state = str(row["Current State"]).strip().upper()
        event = str(row["End Event"]).strip()

        metric_lookup[(state, event)].append(
            {
                "name": row["Metric Name"],
                "type": row["Metric Type"],
                "export": True,
            }
        )

    return metric_lookup


def build_state_machine(
    template_name, station_id, transitions_df, metrics_df, hardware_df
):

    states_dict = defaultdict(lambda: {"transitions": []})
    initial_state = None

    metric_lookup = build_metric_lookup(metrics_df, template_name)

    for _, row in transitions_df.iterrows():
        current_state = row["Current State"]
        event = row["Event (Trigger)"]
        next_state = row["Next State"]

        if pd.isna(current_state) or pd.isna(event) or pd.isna(next_state):
            continue

        current_state = str(current_state).strip().upper()
        next_state = str(next_state).strip().upper()

        if initial_state is None:
            initial_state = current_state

        transition = {
            "event": event,
            "next_state": next_state,
        }

        # ---- attach metrics to transition ----
        metrics = metric_lookup.get((current_state, event))
        if metrics:
            transition["metrics"] = metrics

        states_dict[current_state]["transitions"].append(transition)

    return {
        "initial_state": initial_state or "IDLE",
        "states": dict(states_dict),
    }


# semi irrelevant for now
def parse_state_inference(
    station_id: str, signal_description: str, hardware_df: pd.DataFrame
) -> Optional[Dict[str, bool]]:
    """Parse state inference from signal description"""

    station_hw = hardware_df[hardware_df["Station ID"] == station_id]

    for _, hw_row in station_hw.iterrows():
        hw_tag = hw_row["Hardware Tag / Variable"]

        if signal_description in str(hw_tag):
            category = hw_row["Category"].lower()
            function = hw_row["Function (Role)"]

            key = f"{category}.{function}"

            # Heuristic for expected value
            if "presence" in function.lower():
                return {key: True}
            elif "up" in function.lower() or "lock" in function.lower():
                return {key: True}
            elif "down" in function.lower() or "unlock" in function.lower():
                return {key: False}
            else:
                return {key: True}  # Default

    return None


def _build_base_patterns(event_log_df):
    """
    Build base pattern objects from Event Log Mapping sheet
    key = (target_id, event_type)
    """

    patterns = {}

    for _, row in event_log_df.iterrows():
        target_id = str(row.iloc[0]).strip()
        event_type = str(row.iloc[1]).strip()

        if not event_type or event_type == "nan":
            continue

        regex = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""

        key = (target_id, event_type)

        patterns[key] = {
            "name": event_type.replace("_", " ").title(),
            "regex": regex,
            "event_type": event_type,
            "target_id": target_id,
        }

    return patterns


def _attach_resolvers(patterns, tags_df):
    """
    Enrich patterns with value_mapping and state_resolver from Log Tags sheet
    """

    for _, row in tags_df.iterrows():

        logic_template = str(row.get("logic_template", "")).strip()
        event_type = str(row.get("event_type", "")).strip()
        resolver = str(row.get("event_resolver", "")).strip()

        key = (logic_template, event_type)

        if key not in patterns:
            continue

        pattern = patterns[key]

        # ---------------- VALUE MAPPING ----------------
        if resolver == "value_mapping":
            mapping_key = str(row.get("resolver_type", "")).strip()
            item_key = str(row.get("item_key", "")).strip()
            item_value = str(row.get("item_value", "")).strip()

            if mapping_key:
                pattern.setdefault("value_mapping", {})
                pattern["value_mapping"].setdefault(mapping_key, {})
                pattern["value_mapping"][mapping_key][item_key] = item_value

        # ---------------- STATE RESOLVER ----------------
        elif resolver == "state_resolver":
            item_key = str(row.get("item_key", "")).strip()
            item_value = str(row.get("item_value", "")).strip()

            if item_key:
                pattern.setdefault("state_resolver", {})
                pattern["state_resolver"][item_key] = item_value

    return patterns


def build_event_log_mapping(excel_file: str) -> Dict[str, Any]:
    """Build event_log_mapping.yaml from Excel (patterns based)"""

    event_log_df = pd.read_excel(excel_file, sheet_name="Event Log Mapping")
    tags_df = pd.read_excel(excel_file, sheet_name="Log Tags")

    # Step 1: base pattern definitions
    patterns = _build_base_patterns(event_log_df)

    # Step 2: attach resolvers (value_mapping / state_resolver)
    patterns = _attach_resolvers(patterns, tags_df)

    # Step 3: convert dict → ordered list
    pattern_list = list(patterns.values())

    return {
        "version": "2.0",
        "description": "Auto-generated log parsing patterns",
        "patterns": pattern_list,
    }


def convert_excel_to_yaml(excel_file: str, output_dir: str = "."):
    """Main conversion function"""

    import os

    print("\n" + "=" * 80)
    print("CONVERTING EXCEL TO YAML")
    print("=" * 80)
    print(f"Input: {excel_file}")
    print(f"Output: {output_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build machine_graph.yaml
    print("\n[1/3] Building machine_graph.yaml...")
    machine_graph = build_machine_graph(excel_file)
    machine_graph_path = os.path.join(output_dir, "machine_graph.yaml")

    with open(machine_graph_path, "w", encoding="utf-8") as f:
        yaml.dump(
            machine_graph,
            f,
            Dumper=QuotedStringDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"✓ Wrote {machine_graph_path}")
    print(f"  - {len(machine_graph['topology'])} stations")

    # Build process_logic.yaml
    print("\n[2/3] Building process_logic.yaml...")
    process_logic = build_process_logic(excel_file)
    process_logic_path = os.path.join(output_dir, "process_logic.yaml")

    with open(process_logic_path, "w", encoding="utf-8") as f:
        yaml.dump(
            process_logic,
            f,
            Dumper=QuotedStringDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"✓ Wrote {process_logic_path}")
    print(f"  - {len(process_logic['definitions'])} logic templates")
    #
    # TODO: EVENT LOG MAPPING
    #
    # Build event_log_mapping.yaml
    print("\n[3/3] Building event_log_mapping.yaml...")

    event_mapping = build_event_log_mapping(excel_file)
    event_mapping_path = os.path.join(output_dir, "event_log_mapping.yaml")

    with open(event_mapping_path, "w", encoding="utf=8") as f:
        yaml.dump(
            event_mapping,
            f,
            Dumper=QuotedStringDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"✓ Wrote {event_mapping_path}")
    print(f"  - {len(event_mapping['patterns'])} patterns")
    #
    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE")
    print("=" * 80)

    return {
        "machine_graph": machine_graph_path,
        "process_logic": process_logic_path,
        "event_log_mapping": event_mapping_path,
    }


if __name__ == "__main__":

    # graph = build_machine_graph(
    #     excel_file="~/code/asm-predictive-maintenance/machine_state/GDM_process_details_v2.xlsx"
    # )
    #
    # print(graph)

    # claude combined usage
    # Example usage
    output_files = convert_excel_to_yaml(
        excel_file="../data/CED_process_details_v1.xlsx",
        output_dir="../data/",
    )

    print("\nGenerated files:")
    for name, path in output_files.items():
        print(f"  - {path}")
