import yaml
import re
import os

LOG_FILE = "data/machine_logs.txt"
PATTERN_FILE = "config/log_patterns/test_patterns.yaml"
GRAPH_FILE = "config/machine_graph.yaml"
VERBOSE = True # Turn this on to see every attempt

def load_yaml(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def run_trace():
    print("=== DIGITAL TWIN HANDOVER TRACE (VERBOSE DEBUG) ===")
    patterns = load_yaml(PATTERN_FILE)['patterns']
    graph = load_yaml(GRAPH_FILE)
    topology = {n['id']: n.get('next_nodes', []) for n in graph['topology']}
    inboxes = {n['id']: None for n in graph['topology']}
    active_pallets = {n['id']: None for n in graph['topology']}

    compiled = []
    for p in patterns:
        compiled.append({
            'name': p['name'],
            're': re.compile(p['regex']),
            'event': p['event_type'],
            'target': p.get('target_id'),
            'category': p.get('category', ''),
            'mapping': p.get('value_mapping', {})
        })

    with open(LOG_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or "INFO" not in line: continue
            
            # Extract the message part after the "service - "
            msg_part = line.split(" - ")[-1]
            
            match = None
            pattern = None
            for p in compiled:
                m = p['re'].search(msg_part)
                if m:
                    match, pattern = m, p
                    break
            
            if not match:
                if VERBOSE and "Pallet" in msg_part:
                    print(f"DEBUG Line {line_num}: No match for msg: '{msg_part}'")
                continue

            payload = match.groupdict()
            event = pattern['event']
            category = pattern['category']
            target = pattern['target']
            
            if VERBOSE:
                print(f"DEBUG Line {line_num}: Matched '{pattern['name']}' | Payload: {payload}")

            # 1. Resolve Target via Tag if not hardcoded
            if not target:
                tag = payload.get('tag_id')
                if tag:
                    for n in graph['topology']:
                        if tag in str(n.get('hardware_link', {})):
                            target = n['id']
                            break
            
            if not target: continue

            # 2. IDENTIFICATION
            if 'pallet_id' in payload:
                active_pallets[target] = payload['pallet_id']
                print(f"🆔 [{target}] IDENTIFIED: {payload['pallet_id']}")

            # 3. ARRIVAL
            if category == "arrival":
                node_config = next((n for n in graph['topology'] if n['id'] == target), {})
                if node_config.get('is_entry'):
                    print(f"📍 [{target}] ENTRY POINT: Waiting for scan...")
                elif inboxes.get(target):
                    active_pallets[target] = inboxes[target]
                    inboxes[target] = None
                    print(f"✅ [{target}] CLAIMED FROM INBOX: {active_pallets[target]}")
                else:
                    print(f"❌ [{target}] ARRIVAL FAIL: No Pallet in Inbox!")

            # 4. HANDOVER (The "Push" Logic)
            elif category == "handover":
                pid = active_pallets.get(target)
                
                # If it's the Shuttle routing event, get destination from log
                if event == "PALLET_ROUTE":
                    dest_raw = payload.get('target')
                    dest_map = pattern.get('mapping', {}).get('target', {})
                    dest_id = dest_map.get(dest_raw, dest_raw.lower() if dest_raw else None)
                # Otherwise (Buffers/Dispensers), get destination from Topology
                else:
                    next_nodes = topology.get(target, [])
                    dest_id = next_nodes[0] if len(next_nodes) == 1 else None

                if pid and dest_id:
                    print(f"🚚 [{target}] HANDOVER INITIATED: {pid} -> {dest_id}")
                    inboxes[dest_id] = pid
                else:
                    print(f"⚠️ [{target}] HANDOVER FAILED: ID={pid}, To={dest_id}")

if __name__ == "__main__":
    run_trace()