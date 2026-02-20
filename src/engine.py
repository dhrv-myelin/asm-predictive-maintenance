"""
src/engine.py
The Brain. Manages state transitions, inferences, and metric generation.
Streams directly to PostgreSQL using SQLAlchemy.
"""

import time
from datetime import datetime, timezone
from src.db.models import ProcessMetric, ErrorLog
from src.database import SessionLocal
from src.db.models import ProcessMetric
from src.db.models import ErrorLog
class LogicEngine:
    def __init__(
        self,
        inventory,
        output_file=None,
        visualizer=None,
        tag_resolver=None,
        db_manager=None
    ):
        self.inventory = inventory
        self.output_file = output_file
        self.viz = visualizer
        self.tag_resolver = tag_resolver
        self.db_manager = db_manager

        self.throughput_count = 0
        self.virtual_state_map = {cid: {} for cid in self.inventory}


   # def _stream_error(self, timestamp, event):
      #  try:
          #  ts_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

           # with SessionLocal() as session:
              #  error = ErrorLog(
             #   timestamp=ts_dt,
             #   level=event.get("level"),
             #   message=event.get("raw_line")
          #  )
            #    session.add(error)
             #   session.commit()

           # print(f"[DB] ✓ ERROR logged")

        #except Exception as e:
            #print(f"[DB ERROR - ERROR_LOG] {e}")

 
    def _stream_error(self, timestamp, event):
        try:
            ts_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
 
            with SessionLocal() as session:
                error = ErrorLog(
                    timestamp=ts_dt,
                    severity=event.get("level"),        # Fixed: severity
                    log_message=event.get("raw_line")   # Fixed: log_message
                )
                session.add(error)
                session.commit()
 
            print(f"[DB] ✓ ERROR logged")
 
        except Exception as e:
            print(f"[DB ERROR - ERROR_LOG] {e}")

    # ============================================================
    # EVENT PROCESSING
    # ============================================================

    def process_event(self, timestamp, event):
        payload = event.get("payload", {})
        target_id = event.get("target")
        if event["type"] == "ERROR_LOG":
            print("ENGINE SAW ERROR_LOG")
            self._stream_error(timestamp, event)
            return

        if event["type"] == "SYSTEM_RESET":
            self._reset_system()
            print("\n[RESET] Throughput counter reset.\n")

        # 1. Resolve Target (Tag -> ID)
        if not target_id and 'tag_id' in payload and self.tag_resolver:
            target_id = self.tag_resolver.resolve(payload['tag_id'])
            event['target'] = target_id
            print(f"[DEBUG]: Resolved Tag '{payload['tag_id']}' to Target ID: {target_id}")
        
        if not target_id and 'state_resolver' in event:
            state_resolver = event.get('state_resolver', {})
            # print(f"[DEBUG]: state_resolver: {state_resolver}")
            target_id = self._state_based_target_resolver(state_resolver, payload)
            print(f"[DEBUG]: State Resolved target from {state_resolver} to Target ID: {target_id}" )
        
        if target_id=="system":
            print("event : ",event)
            self._push_raw_metrics(event,timestamp)
        payload = event.get("payload", {})
        target_id = event.get("target")
        if event["type"] == "ERROR_LOG":
            print("ENGINE SAW ERROR_LOG")
            self._stream_error(timestamp, event)
            return

        if event["type"] == "SYSTEM_RESET":
            self.throughput_count = 0
            print("\n[RESET] Throughput counter reset.\n")

        # Resolve tag → station
        if not target_id and "tag_id" in payload and self.tag_resolver:
            target_id = self.tag_resolver.resolve(payload["tag_id"])
            event["target"] = target_id

        if not target_id and "state_resolver" in event:
            target_id = self._state_based_target_resolver(
                event.get("state_resolver", {}), payload
            )

        if target_id == "system":
            self._push_raw_metrics(event, timestamp)
            return

        if not target_id or target_id not in self.inventory:
            return

        station = self.inventory[target_id]
        transition_stage = ""

        for item in (
            station.logic_template.get("states", {})
            .get(station.current_state, {})
            .get("transitions", [])
        ):
            if item.get("event") == event["type"]:
                transition_stage = item.get("transition_stage", "")

        if "pallet_id" in payload:
            station.active_pallet_id = payload["pallet_id"]

        # Arrival
        if transition_stage == "arrival" and not getattr(station, "is_entry", False):
            if station.expected_pallet_id:
                station.active_pallet_id = station.expected_pallet_id
                station.expected_pallet_id = None

        # Handover prep
        if transition_stage == "handover_prep" and station.active_pallet_id:
            dest_id = self._get_destination_id(station, event)
            if dest_id:
                self._prepare_handover(station, dest_id)

        # Core state machine
        self._handle_station_logic(station, event, timestamp)

        # Execute handover
        if transition_stage == "handover":
            self._execute_handover(station, event)

        # Visualization
        if self.viz:
            context = {"pallet_id": station.active_pallet_id}
            self.viz.update(timestamp, station.id, station.current_state, context)

    def _reset_system(self):
        """
        Resets all mutable engine and station state to initial values.
        Called when a SYSTEM_RESET event is received.
        """
        # 1. Reset engine-level counters
        self.throughput_count = 0

        # 2. Clear all virtual sensor memory
        self.virtual_state_map = {cid: {} for cid in self.inventory}

        # 3. Reset every station back to its configured initial state
        for station in self.inventory.values():
            station.current_state = station.logic_template.get('initial_state', 'IDLE')
            station.previous_state = None
            station.state_entry_time = 0.0
            station.active_pallet_id = None
            station.expected_pallet_id = None
            station.current_destination = None
            station.metric_timers = {}

        print("--------------------------------------------")
        print("[SYSTEM RESET]: All station states, pallet tracking, and counters cleared.")
        print("--------------------------------------------")

    def _get_destination_id(self, station, event):
        payload = event.get("payload", {})

        if "destination" in payload:
            return payload["destination"]

        nodes = getattr(station, "downstream_nodes", [])
        if len(nodes) == 1:
            return nodes[0]

        return None

    def _prepare_handover(self, station, dest_id):
        self.inventory[dest_id].expected_pallet_id = station.active_pallet_id
        station.current_destination = dest_id

    def _execute_handover(self, station, event):
        destination_id = station.current_destination or "exit_sink"

        if self.inventory[destination_id].is_exit:
            self._record_completion(
                self.inventory[destination_id],
                station.active_pallet_id,
                event.get("timestamp", time.time()),
            )

        station.active_pallet_id = None
        station.current_destination = None

    def _record_completion(self, station, pallet_id, timestamp):
        self.throughput_count += 1

        self._stream_metric(
            timestamp=timestamp,
            comp_id=station.id,
            name="throughput_total",
            value=self.throughput_count,
            unit="units",
            context="EXIT",
        )

        if self.viz:
            self.viz.log_completion(timestamp, self.throughput_count)
            self.viz.update_scoreboard(self.throughput_count)

    # ============================================================
    # STATE MACHINE
    # ============================================================

    def _handle_station_logic(self, station, event, timestamp):
        current_state_def = station.logic_template["states"].get(
            station.current_state
        )
        if not current_state_def:
            return False

        for trans in current_state_def.get("transitions", []):
            if trans["event"] == event["type"]:
                if self._check_hardware_prereqs(station, trans):
                    self._execute_transition(station, trans, timestamp)
                    return True

        return False

    def _execute_transition(self, station, transition, timestamp):
        curr_def = station.logic_template["states"][station.current_state]
        self._generate_metrics(station, curr_def, transition, timestamp)

        if "state_inference" in transition:
            self._apply_inference(station, transition["state_inference"])

        station.set_state(transition["next_state"], timestamp)

    def _check_hardware_prereqs(self, station, transition):
        checks = transition.get("hardware_check", [])
        if not checks:
            return True

        virtual_memory = self.virtual_state_map[station.id]

        for condition in checks:
            if "==" not in condition:
                continue

            key, val_str = condition.split("==")
            key = key.strip()
            expected_val = val_str.strip().lower() == "true"
            actual_val = virtual_memory.get(key, False)

            if actual_val != expected_val:
                return False

        return True

    def _apply_inference(self, station, inference_dict):
        for key, value in inference_dict.items():
            self.virtual_state_map[station.id][key] = value

    # ============================================================
    # METRICS
    # ============================================================

    def _generate_metrics(self, station, state_def, transition, timestamp):
        metrics_config = []

        for trans in state_def.get("transitions", []):
            if trans["event"] == transition["event"]:
                metrics_config = trans.get("metrics", [])
                break

        for m_conf in metrics_config:
            m_name = m_conf["name"]
            m_type = m_conf["type"]
            value = None

            if m_type == "duration_seconds":
                if station.state_entry_time > 0:
                    value = timestamp - station.state_entry_time

            if value is not None:
                self._stream_metric(
                    timestamp,
                    station.id,
                    m_name,
                    value,
                    m_type,
                    station.current_state,
                )

    def _push_raw_metrics(self, event, timestamp):
        payload = event.get("payload", {})

        for key, value in payload.items():
            if key == "pallet_id":
                continue

            self._stream_metric(
                timestamp=timestamp,
                comp_id=event.get("target", "system"),
                name=key,
                value=value,
                unit=None,
                context=event.get("type"),
            )

    # ============================================================
    # DATABASE STREAM
    # ============================================================

    def _stream_metric(self, timestamp, comp_id, name, value, unit, context):
        try:
            #ts_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            ts_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            with SessionLocal() as session:
                metric = ProcessMetric(
                    timestamp=ts_dt,
                    station_name=comp_id,
                    metric_name=name,
                    value=round(value, 4)
                    if isinstance(value, (int, float))
                    else value,
                    unit=unit,
                    state_context=context,
                )
                session.add(metric)
                session.commit()

            print(f"[DB] ✓ {comp_id}.{name} = {value}")

        except Exception as e:
            print(f"[DB ERROR] {e}")

    # ============================================================
    # TARGET RESOLUTION
    # ============================================================

    def _state_based_target_resolver(self, state_resolver, payload):
        target_type = state_resolver.get("target_type")
        current_state = state_resolver.get("current_state")
        logic_type = state_resolver.get("logic_type")

        if logic_type == "filter_expected_pallet_id_state":
            for station_id, station in self.inventory.items():
                if (
                    station.current_state == current_state
                    and station.expected_pallet_id == payload.get("pallet_id")
                ):
                    return station_id

        if logic_type == "filter_station_type_state_oldest":
            oldest = None
            for station_id, station in self.inventory.items():
                if (
                    station.config.get("logic_template") == target_type
                    and station.current_state == current_state
                ):
                    if not oldest or station.state_entry_time < oldest[1]:
                        oldest = (station_id, station.state_entry_time)

            if oldest:
                return oldest[0]

        return None
