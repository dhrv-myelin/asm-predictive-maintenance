"""
src/log_parser.py
Parses industrial logs. Supports Static Analysis and Live Tailing.
"""
import re
import time
import yaml
from datetime import datetime

class LogParser:
    def __init__(self, patterns_config_path):
        self.patterns = []
        self._load_patterns(patterns_config_path)
        
        # Header Regex: Timestamp | Metadata/Service | Message
        self.header_pattern = re.compile(
            r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[\.,]\d{3})\s+(.*)"
        )
        self.service_extractor = re.compile(r".*?\s+(\w+)\s+-\s+.*")

    def _load_patterns(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        for p in data['patterns']:
            if p.get('regex') is None: continue
            try:
                self.patterns.append({
                    'name': p['name'],
                    'regex': re.compile(p['regex']),
                    'event_type': p['event_type'],
                    'target_id': p.get('target_id'),
                    'state_resolver': p.get('state_resolver', {}),
                    'mapping': p.get('value_mapping', {})
                })
            except re.error as e:
                print(f"Error compiling regex for '{p['name']}': {e}")

    def parse_file(self, log_path, live_mode=False):
        """
        Generator: Yields (timestamp_float, event_dict).
        If live_mode=True, it never returns; it keeps waiting for new lines.
        """
        print(f"Parsing: {log_path} (Live Mode: {live_mode})")
        
        with open(log_path, 'r') as f:
            # 1. Catch Up Phase (Read existing data)
            while True:
                line = f.readline()
                if not line:
                    if live_mode:
                        # End of file reached, but we are live.
                        # Wait briefly and try reading again.
                        time.sleep(0.1) 
                        continue
                    else:
                        # Static mode: End of file means we are done.
                        break
                
                # Process the valid line
                clean_line = line.strip()
                if not clean_line: continue
                
                yield from self._process_line(clean_line, line)

    def _process_line(self, line_text, full_raw_line):
        """Helper to parse a single line and yield events if matched."""
        
        print("--------------new event--------------")
        # 1. Extract Timestamp
        match_header = self.header_pattern.match(line_text)
        if not match_header:
            return

        ts_str = match_header.group(1)
        content_to_match = match_header.group(2)

        try:
            dt = datetime.strptime(ts_str.replace(',', '.'), "%Y-%m-%d %H:%M:%S.%f")
            epoch = dt.timestamp()
        except ValueError:
            return

        if ' ERROR ' in line_text:
            yield epoch, {
                "type": "ERROR_LOG",
                "target": None,
                "level": "ERROR",
                "destination": None,
                "category": "",
                "payload": {},
                "raw_line": line_text
            }
            return
        elif ' WARN ' in line_text:
            yield epoch, {
                "type": "WARN_LOG",
                "target": None,
                "level": "WARN",
                "destination": None,
                "category": "",
                "payload": {},
                "raw_line": line_text
            }
            return
        svc_match = self.service_extractor.match(content_to_match)
        service_name = svc_match.group(1) if svc_match else "Unknown"

        # 3. Match Logic Regex
        for pattern in self.patterns:
            pmatch = pattern['regex'].search(content_to_match)
            if pmatch:
                # We yield so the loop continues
                print(f"[DEBUG]: Matching for Log Line: {content_to_match}")
                yield epoch, self._build_event(pattern, pmatch, service_name)
                break

    def _build_event(self, pattern, match, service_name, raw_line=""):
        groups = match.groupdict()
        
        # 1. Apply Mappings to ALL capture groups first
        mapped_payload = groups.copy()
        for key, value in mapped_payload.items():
            if key in pattern.get('mapping', {}):
                mapped_payload[key] = pattern['mapping'][key].get(value, value)

        # 2. Get "Event Owner" (The Target)
        target = pattern.get('target_id')
        print(f"[DEBUG]: Event matched for pattern '{pattern['name']}' with hardcoded target_id: {target}")

        if not target and 'target' in mapped_payload:
            target = mapped_payload.get('target')
            print(f"[DEBUG]: Event matched for pattern '{pattern['name']}' with dynamic target from payload: {target}")

        # 3. Handle Handover Destination (The "Pull" signal)
        # We explicitly keep 'destination' in the payload for the Engine to use
        destination = mapped_payload.get('destination')

        return {
            "type": pattern['event_type'],
            "target": target,  
            "level": service_name,    # Who reported the log
            "destination": destination, # Where the pallet is going (if known)
            "state_resolver" : pattern.get('state_resolver', {}), # Optional instructions for state inference
            "payload": mapped_payload,
            "raw_line": raw_line  # Store the complete log line
        }
