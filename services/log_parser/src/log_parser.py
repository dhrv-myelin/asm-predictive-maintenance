import re
import yaml
from datetime import datetime, date


# ─────────────────────────────────────────────
#  BASE ADAPTER  (defines the contract)
# ─────────────────────────────────────────────
class MachineAdapter:
    """
    Every machine adapter must implement:
      • extract_header(line)  → (epoch, channel, content) | None
    Everything else (pattern matching, _build_event) lives in LogParser.
    """

    def extract_header(self, line: str):
        """
        Parse the very first line of a log entry.
        Returns (epoch_float, channel_str, content_str) or None if line
        doesn't look like a log header.
        """
        raise NotImplementedError


# ─────────────────────────────────────────────
#  GDM ADAPTER
#  Format: 2024-01-15 08:05:25.554 [LEVEL] Service - message
# ─────────────────────────────────────────────
class GDMAdapter(MachineAdapter):
    _HEADER = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[\.,]\d{3})\s+(.*)"
    )
    _SERVICE = re.compile(r".*?\s+(\w+)\s+-\s+.*")

    def __init__(self, log_date: str = None):
        pass  # GDM has full timestamps — log_date is unused

    def extract_header(self, line: str):
        m = self._HEADER.match(line)
        if not m:
            return None

        ts_str, content = m.group(1), m.group(2)
        try:
            dt = datetime.strptime(ts_str.replace(',', '.'), "%Y-%m-%d %H:%M:%S.%f")
            epoch = dt.timestamp()
        except ValueError:
            return None

        svc_m = self._SERVICE.match(content)
        channel = svc_m.group(1) if svc_m else "Unknown"

        return epoch, channel, content


# ─────────────────────────────────────────────
#  CED ADAPTER
#  Format: CHANNEL————————>HH:MM:SS.mmm  message
#  Multi-line block starts with  给PDCA发送:_{  and ends with  }
# ─────────────────────────────────────────────
class CEDAdapter(MachineAdapter):
    # Matches:  PUBLIC————————>08:05:25.554  some text
    _HEADER = re.compile(
        r"^(\w+)[-—>]+(\d{2}:\d{2}:\d{2}[.,]\d+)\s+(.*)"
    )
    # Detects the opening of a  给PDCA发送:_{  block in the message part
    _PDCA_BLOCK_OPEN = re.compile(r"给PDCA发送:_\{")
    _SFC_OK_BLOCK_OPEN = re.compile(r"ok@\{0 SFC_OK$")
    # One data row inside a  给PDCA发送:_{...}  block
    # e.g.  J63HQG006TB0000WDH@pdata@glue_weight@10@9.5@10.5@mg
    _DATA_ROW = re.compile(
        r"^(?P<serial>\w+)"
        r"@(?P<record_type>dut_pos|pdata|attr|submit)"
        r"(?:@(?P<rest>.*))?$"
    )
    # Add this to capture @start lines of the PDCA Group : r"@(?P<record_type>start|dut_pos|pdata|attr|submit)"

    def __init__(self, log_date: str):
        # log_date arrives as "YYYY-MM-DD" string (already formatted by caller)
        self._log_date = log_date  # e.g. "2024-01-15"

    def extract_header(self, line: str):
        m = self._HEADER.match(line)
        if not m:
            return None

        channel, ts_str, content = m.group(1), m.group(2), m.group(3).strip()

        time_part = ts_str.replace(',', '.')
        hms, frac = time_part.split('.')
        frac = frac.ljust(3, '0')[:3]
        full_ts = f"{self._log_date} {hms}.{frac}"
        try:
            dt = datetime.strptime(full_ts, "%Y-%m-%d %H:%M:%S.%f")
            epoch = dt.timestamp()
        except ValueError:
            return None

        return epoch, channel, content

    def parse_data_row(self, raw_row: str) -> dict | None:
        """
        Parse one @-separated data row from a 给PDCA发送:_{...} block.
        Returns a structured dict or None if the row doesn't match.
        """
        m = self._DATA_ROW.match(raw_row.strip())
        if not m:
            return None

        serial      = m.group('serial')
        record_type = m.group('record_type')
        rest        = m.group('rest') or ''
        parts       = rest.split('@') if rest else []

        if record_type == 'dut_pos':
            return {
                'pallet_id':      serial,
                'record_type':   'dut_pos',
                'carrier_sn':  parts[0] if len(parts) > 0 else None,
                'position':    parts[1] if len(parts) > 1 else None,
            }

        if record_type == 'pdata':
            # Determine schema by whether lower/upper limits are present
            # No-limit fields have empty strings at positions 2 and 3: @value@@@
            has_limits = (
                len(parts) > 3
                and parts[2] != ''   # lower is not empty
                and parts[3] != ''   # upper is not empty
            )

            if has_limits:
                # e.g. gantry_cpk_x@0.01@-0.05@0.05@mm
                #      glue_weight@10@9.5@10.5@mg
                #      nozzle_temp@37@35@45@degrees
                return {
                    'pallet_id':      serial,
                    'record_type':   'pdata',
                    f'{parts[0]}_value':   parts[1] if len(parts) > 1 else None,
                    f'{parts[0]}_lower':   parts[2] if len(parts) > 2 else None,
                    f'{parts[0]}_upper':   parts[3] if len(parts) > 3 else None,
                    'unit':        parts[4] if len(parts) > 4 else None,
                }
            else:
                # e.g. cm_vendor@0@@@
                #      pulse@1@@@
                #      operator_id@1@@@
                return {
                    'pallet_id':      serial,
                    'record_type':   'pdata',
                    f'{parts[0]}_value':       parts[1] if len(parts) > 1 else None,
                    'unit':        parts[4] if len(parts) > 4 else None,  # e.g. cycle_time@17.859@@@s
                }

        if record_type == 'attr':
            return {
                'pallet_id':      serial,
                'record_type': 'attr',
                f'{parts[0]}':   parts[1] if len(parts) > 1 else None,
            }

        if record_type == 'submit':
            return {
                'pallet_id':      serial,
                'record_type': 'submit',
                'version':     parts[0] if len(parts) > 0 else None,
            }

        return None


# ─────────────────────────────────────────────
#  UNIFIED  LOG  PARSER
# ─────────────────────────────────────────────
class LogParser:
    """
    Machine-agnostic parser. Pass machine='GDM' or machine='CED' at construction.
    _build_event always returns the same dict shape regardless of machine.

    CED multi-line blocks are handled statelessly across process_line() calls —
    no peek_fn or process_file() needed. The main loop can call process_line()
    one line at a time and blocks are assembled internally.
    """

    MACHINE_ADAPTERS = {
        'GDM': GDMAdapter,
        'CED': CEDAdapter,
    }

    def __init__(self, patterns_config_path: str, machine: str, log_date: str):
        if machine not in self.MACHINE_ADAPTERS:
            raise ValueError(f"Unknown machine '{machine}'. Choose from {list(self.MACHINE_ADAPTERS)}")

        self.adapter: MachineAdapter = self.MACHINE_ADAPTERS[machine](log_date=log_date)
        self.machine = machine
        self.patterns: list[dict] = []
        self._load_patterns(patterns_config_path)

        # ── CED stateful block assembly ───────────────────────────────────
        # These track state across successive process_line() calls so that
        # the main loop never needs to know about multi-line blocks at all.
        self._in_block: bool       = False   # are we currently inside a _{ } block?
        self._block_lines: list    = []      # accumulating rows for current block
        self._block_epoch: float   = 0.0     # epoch of the opener line
        self._block_channel: str   = ''      # channel of the opener line

    # ── pattern loading ────────────────────────────────────────────────────
    def _load_patterns(self, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        for p in data.get('patterns', []):
            if p.get('regex') is None:
                continue
            try:
                self.patterns.append({
                    'name':           p['name'],
                    'regex':          re.compile(p['regex']),
                    'event_type':     p['event_type'],
                    'target_id':      p.get('target_id'),
                    'state_resolver': p.get('state_resolver', {}),
                    'mapping':        p.get('value_mapping', {}),
                })
            except re.error as e:
                print(f"Error compiling regex for '{p['name']}': {e}")

    # ── public API ─────────────────────────────────────────────────────────
    def process_line(self, line_text: str, peek_fn=None):
        """
        Parse one log line. Yields (epoch, event_dict) tuples.

        For CED machines, multi-line 给PDCA发送:_{ ... } blocks are assembled
        across successive calls — no peek_fn or special handling needed from
        the caller. peek_fn is accepted but unused (kept for API compatibility).
        """
        if self.machine == 'CED':
            yield from self._process_line_ced(line_text)
        else:
            yield from self._process_line_gdm(line_text)

    # ── GDM path ───────────────────────────────────────────────────────────
    def _process_line_gdm(self, line_text: str):
        parsed = self.adapter.extract_header(line_text)
        if parsed is None:
            return

        epoch, channel, content = parsed

        if '] ERROR ' in line_text:
            yield epoch, self._sentinel_event("ERROR_LOG", "ERROR", line_text)
            return
        if '] WARN  ' in line_text:
            yield epoch, self._sentinel_event("WARN_LOG", "WARN", line_text)
            return

        for pattern in self.patterns:
            pmatch = pattern['regex'].search(content)
            if pmatch:
                yield epoch, self._build_event(pattern, pmatch, channel, line_text)
                break

    # ── CED path ───────────────────────────────────────────────────────────
    def _process_line_ced(self, line_text: str):
        adapter: CEDAdapter = self.adapter  # type: ignore[assignment]
        stripped = line_text.strip()

        # ── Case 1: we are already inside a _{ ... } block ────────────────
        if self._in_block:
            if re.search(r"OK:(?P<payload>.*)\}@", stripped):

                print("[DEBUG] Found SFC_OK Block close")

                self._in_block = False

                for pattern in self.patterns:
                    pmatch = pattern['regex'].search(stripped)
                    # print(f"stripped ::{len(self._block_lines)}:: {stripped}")
                    if pmatch:
                        print("[DEBUG] Matched Regex for SFC_OK Block close, Building Event")
                        yield self._block_epoch, self._build_event(pattern, pmatch, self._block_channel, line_text)
                        self._block_lines   = []
                        self._block_channel = ''
                        self._block_epoch   = 0.0
                        break

            elif stripped == '}':
                # Closing brace — block is complete, fire all events
                self._in_block = False
                yield from self._process_ced_block(
                    self._block_epoch,
                    self._block_channel,
                    self._block_lines,
                )
                self._block_lines   = []
                self._block_channel = ''
                self._block_epoch   = 0.0
            else:
                # Still accumulating rows — do nothing else this call
                self._block_lines.append(stripped)
            return

        # ── Case 2: normal line — parse header first ───────────────────────
        parsed = adapter.extract_header(line_text)
        if parsed is None:
            return

        epoch, channel, content = parsed

        # ── Case 3: block opener ───────────────────────────────────────────
        if adapter._PDCA_BLOCK_OPEN.search(line_text):
            self._in_block      = True
            self._block_epoch   = epoch
            self._block_channel = channel
            self._block_lines   = [line_text]    # index 0 = opener
            return                               # rows will arrive in future calls

        if adapter._SFC_OK_BLOCK_OPEN.search(content):
            print("[DEBUG] Found SFC_OK Block open")
            self._in_block      = True
            self._block_epoch   = epoch
            self._block_channel = channel
            self._block_lines   = [line_text]    # index 0 = opener
            return                               # rows will arrive in future calls

        # ── Case 4: regular single-line CED event — pattern match ──────────
        for pattern in self.patterns:
            pmatch = pattern['regex'].search(content)
            if pmatch:
                yield epoch, self._build_event(pattern, pmatch, channel, line_text)
                break

    # ── CED block exploder ─────────────────────────────────────────────────
    def _process_ced_block(self, epoch: float, channel: str, block_lines: list[str]):
        """
        Explode a 给PDCA发送:_{...} block into one normalised event per data row.
        block_lines[0] is the opener; block_lines[1:] are the raw data rows.
        """
        adapter: CEDAdapter = self.adapter  # type: ignore[assignment]

        event_type_map = {
            'start':   'CED_START',
            'dut_pos': 'CED_DUT_POS',
            'pdata':   'CED_PDATA',
            'attr':    'CED_ATTR',
            'submit':  'CED_SUBMIT',
        }

        for raw_row in block_lines[1:]:      # skip opener at index 0
            row = adapter.parse_data_row(raw_row)
            if row is None:
                continue

            # serial      = row.get('serial', 'Unknown')
            record_type = row.pop('record_type', 'unknown') # Remove recod_type before passing to payload

            yield epoch, {
                "type":           event_type_map.get(record_type, 'CED_UNKNOWN'),
                "target":         'system',       # serial number = event owner
                "level":          channel,      # PUBLIC / LEFT / RIGHT …
                "destination":    None,
                "state_resolver": {},
                "payload":        row,          # full parsed row dict
                "raw_line":       raw_row,
            }

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _sentinel_event(event_type: str, level: str, raw_line: str) -> dict:
        return {
            "type":           event_type,
            "target":         None,
            "level":          level,
            "destination":    None,
            "state_resolver": {},
            "category":       "",
            "payload":        {},
            "raw_line":       raw_line,
        }

    def _build_event(self, pattern: dict, match, service_name: str, raw_line: str = "") -> dict:
        groups = match.groupdict()

        # Apply value mappings
        mapped_payload = groups.copy()
        for key, value in mapped_payload.items():
            if key in pattern.get('mapping', {}):
                mapped_payload[key] = pattern['mapping'][key].get(value, value)

        # Resolve target
        target = pattern.get('target_id')
        if not target:
            target = mapped_payload.get('target')

        return {
            "type":           pattern['event_type'],
            "target":         target,
            "level":          service_name,
            "destination":    mapped_payload.get('destination'),
            "state_resolver": pattern.get('state_resolver', {}),
            "payload":        mapped_payload,
            "raw_line":       raw_line,
        }