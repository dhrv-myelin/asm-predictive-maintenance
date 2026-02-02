"""
src/engine.py
The Brain. Manages state transitions, inferences, and metric generation.
"""
import csv
import time
from datetime import datetime
from unicodedata import category

class LogicEngine:
    def __init__(self, inventory, output_file="data/process_metrics.csv", visualizer=None, buffer_size=10000, tag_resolver=None):
        self.inventory = inventory  # Dict of {id: Station} from inventory.py
        self.output_file = output_file
        self.viz = visualizer
        self.tag_resolver = tag_resolver
        
        # Performance: Pre-Allocated Buffer for Metrics
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size 
        self.buf_idx = 0
        self.throughput_count = 0
        
        # Virtual Reality Map (simulating the hardware state)
        # Structure: {'l1_dispenser': {'sensors.presence': True, ...}}
        self.virtual_state_map = {cid: {} for cid in self.inventory}

        # Initialize Output File
        self._init_output_file()

    def _init_output_file(self):
        """Creates the file with header if it doesn't exist."""
        try:
            with open(self.output_file, 'x', newline='') as f:
                writer = csv.writer(f)
                # TALL FORMAT: Easy for Pivot Tables & SQL
                writer.writerow(["timestamp", "component_id", "metric_name", "value", "unit", "state_context"])
        except FileExistsError:
            pass # Append mode is fine

    def process_event(self, timestamp, event):
        payload = event.get('payload', {})
        category = event.get('category', '')
        target_id = event.get('target')

        if event['type'] == "SYSTEM_RESET":
            self.throughput_count = 0
            print("--------------------------------------------\n")
            print("[DEBUG]: System Reset Event Processed. Throughput counter reset to 0.")
            print("\n--------------------------------------------")

        print(f"[DEBUG]: Payload: {payload}, Category: {category}, Target ID: {target_id}")

        # 1. Resolve Target (Tag -> ID)
        if not target_id and 'tag_id' in payload and self.tag_resolver:
            target_id = self.tag_resolver.resolve(payload['tag_id'])
            event['target'] = target_id
            print(f"[DEBUG]: Resolved Tag '{payload['tag_id']}' to Target ID: {target_id}")

        if not target_id or target_id not in self.inventory:
            print(f"[DEBUG]: ❌ Event skipped: Unknown target ID '{target_id}'")
            return
        
        station = self.inventory[target_id]

        # 2. IDENTIFY: If a pallet ID is in the log, update the station immediately
        if 'pallet_id' in payload:
            station.active_pallet_id = payload['pallet_id']
            print(f"[DEBUG]: Updated Station '{target_id}' with Pallet ID: {station.active_pallet_id}")

        # 3. ARRIVAL: Claim from Inbox
        if category == "arrival":
            if not getattr(station, 'is_entry', False):
                if station.expected_pallet_id:
                    station.active_pallet_id = station.expected_pallet_id
                    station.expected_pallet_id = None
                    print(f"[DEBUG]: ✅ [{target_id}] ARRIVAL SUCCESS: Claimed Pallet ID {station.active_pallet_id} from Inbox")
                else:
                    # Use standard print - if this doesn't show, the 'arrival' category is missing in YAML
                    print(f"[DEBUG]: ❌ [{target_id}] ARRIVAL FAIL: Inbox Empty")

    
        # 4. HANDOVER PREP: Prepare for Handover to Next Station
        if category == "handover_prep" and station.active_pallet_id:
            destination_id = self._get_destination_id(station, event)
            if destination_id is None:
                print(f"[WARN] ⚠️ [HANDOVER PREP] {station.id}: Buggy handover prep as destination_id could not be determined.")
            self._prepare_handover(station, dest_id=destination_id)

        # 5. LOGIC: Run the actual state machine
        self._handle_station_logic(station, event, timestamp)

        if category == "handover":
            self._execute_handover(station, event)

        # D. Viz Update
        if self.viz:
            print(f"[DEBUG]: Updating Viz for Station {station.id} in State {station.current_state} with Pallet {station.active_pallet_id}")
            context = {'pallet_id': station.active_pallet_id}
            self.viz.update(timestamp, station.id, station.current_state, context)

    def _get_destination_id(self, station, event):
        """Given a station and event, determine the destination station ID."""
        payload = event.get('payload', {})
        dest_id = None

        # A. Explicit Target - A routing node (like the shuttle)
        if 'destination' in payload:
            dest_id = payload['destination']
            print(f"[DEBUG]: GET DESTINATION: Explicit target found: {dest_id}")
        
        # B. Implicit Target (Buffer/Dispenser case - use Graph)
        else:
            nodes = getattr(station, 'downstream_nodes', [])
            if len(nodes) == 1:
                dest_id = nodes[0]
                print(f"[DEBUG]: 🟢 [GET DESTINATION] {station.id}: Single (implicit) downstream node found: {dest_id}")
            else:
                print(f"[WARN] ⚠️ [GET DESTINATION] {station.id}: Cannot determine destination as we have {len(nodes)} downstream nodes inspite of being a non-routing station.")
        
        if dest_id and dest_id not in self.inventory:
            print(f"[WARN] ⚠️ [GET DESTINATION] {station.id}: Destination ID '{dest_id}' is invalid. Setting it to None. Only these stations exist: {list(self.inventory.keys())}")
            dest_id = None

        return dest_id

    def _prepare_handover(self, station, dest_id):
        """Prepares for handover by setting the destination node's expected pallet id. This is essential as the current pallet id could be lost or overwritten if something else is already entering this station. In case of routing stations, essential to set the current_destination as it could be lost post this preparation, so we always set it regardless of routing/non-routing stations. TO BE CALLED ONLY IF CATEGORY == 'handover_prep'"""
        self.inventory[dest_id].expected_pallet_id = station.active_pallet_id
        station.current_destination = dest_id
        print(f"[DEBUG]: 🚚 HANDOVER PREP FOR {station.id} -> {dest_id} | Pallet: {station.active_pallet_id}")

    def _execute_handover(self, station, event):
        """Completes the handover by clearing the active pallet id from the current station and incrementing pallets counter for last station exit. In case of routing stations, clears the current_destination for the station. TO BE CALLED ONLY IF CATEGORY == 'handover'"""
        destination_id = station.current_destination
        if destination_id is None:
            destination_id = "exit_sink" # That's the only valid reason to not have a destination as per graph topology
            print("[DEBUG] No destination_id found during handover, implying no handover prep. That means it is the exit_sink.")
        if self.inventory[destination_id].is_exit:
            # Increment the pallets counter for the last station exit
            self._record_completion(self.inventory[destination_id], station.active_pallet_id, event.get('timestamp', time.time()))
            print(f"[DEBUG]: 🏁 Pallet {station.active_pallet_id} has EXITED the system from {station.id}.")
        # Clear the active pallet ID from the current station
        station.active_pallet_id = None
        # Clear the current_destination
        station.current_destination = None

    def _record_completion(self, station, pallet_id, timestamp):
        # 1. Increment the internal counter
        self.throughput_count += 1
        
        # 2. Add to the Metrics Buffer (The Permanent Record)
        # This allows you to graph "Throughput Over Time" later
        self._add_to_buffer(
            timestamp=timestamp,
            comp_id=station.id,
            name="throughput_total",
            value=self.throughput_count,
            context="EXIT"
        )

        if self.viz:
            # Update the graph
            self.viz.log_completion(timestamp, self.throughput_count)
            # Update the text readout
            self.viz.update_scoreboard(self.throughput_count)
        
        print(f"[DEBUG]: 🏆 TOTAL THROUGHPUT: {self.throughput_count} (Last: {pallet_id})")


    def _handle_station_logic(self, station, event, timestamp):
        """
        Returns True if a valid transition was found and executed.
        """
        current_state_def = station.logic_template['states'].get(station.current_state)
        if not current_state_def: return False

        # 1. Check strict transitions
        for trans in current_state_def.get('transitions', []):
            if trans['event'] == event['type']:
                print(f"[DEBUG]: Attempting transition {trans['event']} for Station {station.id} in State {station.current_state}")
                if self._check_hardware_prereqs(station, trans):
                    self._execute_transition(station, trans, timestamp)
                    return True
        return False

    def _handle_permissive_catchup(self, station, event, timestamp):
        """
        The 'Magic' Fix: Looks ahead 1-2 steps to see if we missed a log.
        Currently not used, but needed in future for robustness.
        """
        curr_state_name = station.current_state
        logic_states = station.logic_template['states']
        
        # brute-force search: Which state *does* accept this event?
        # We look for a state 'target_candidate' that responds to this event
        possible_intermediate_states = []
        
        for state_name, state_def in logic_states.items():
            for trans in state_def.get('transitions', []):
                if trans['event'] == event['type']:
                    # Found a state that WOULD accept this event
                    # Now, is it reachable from current state?
                    # For MVP, we allow a "Force Jump" if we can find a direct path
                    # But often, simplest is best: just FORCE the transition logic to run
                    # as if we were already in that state.
                    
                    # 1. "Fake" the exit from current state (Duration = Unknown/Long)
                    # We accept the gap.
                    
                    # 2. "Teleport" to the state that accepts this event
                    print(f"⚠️ [JUMP] {station.id}: Missed transition. Jumping {curr_state_name} -> {state_name}")
                    
                    # Force update the state without metrics (because we missed the real entry time)
                    station.current_state = state_name
                    station.state_entry_time = timestamp # Reset clock
                    
                    # 3. Now execute the ACTUAL event that brought us here
                    # This ensures we enter the *next* state correctly
                    self._execute_transition(station, trans, timestamp)
                    return

    def _execute_transition(self, station, transition, timestamp):
        """Update the state for the station and generate metrics and inferences."""
        # A. Metrics (Exit Old State)
        curr_def = station.logic_template['states'][station.current_state]
        self._generate_metrics(station, curr_def, timestamp)
        
        # B. Inference
        if 'state_inference' in transition:
            self._apply_inference(station, transition['state_inference'])
            
        # C. Move
        station.set_state(transition['next_state'], timestamp)
        print(f"[DEBUG]: ✅ [{station.id}] Transitioned to State '{station.current_state}' at {datetime.fromtimestamp(timestamp).isoformat()}")


    def _check_hardware_prereqs(self, station, transition):
        """
        Verifies if virtual sensors match requirements.
        Format: ["sensors.presence == True"]
        """
        checks = transition.get('hardware_check', [])
        if not checks: return True

        virtual_memory = self.virtual_state_map[station.id]

        for condition in checks:
            # Simple Parser: "key == value"
            if "==" not in condition: continue
            
            key, val_str = condition.split('==')
            key = key.strip()
            expected_val = val_str.strip().lower() == 'true'
            
            # Check Memory (Default to False if unknown)
            actual_val = virtual_memory.get(key, False)
            
            if actual_val != expected_val:
                return False # Constraint Failed
        
        return True

    def _handle_handover(self, station, event):
        # We only act if the pattern category is 'handover'
        if event.get('category') != "handover":
            return

        print(f"DEBUG: Handover triggered for {station.id}. Current ID: {station.active_pallet_id}")

        # 1. SHUTTLE: Use the mapping to find the destination
        if event.get('type') == "PALLET_ROUTE":
            dest_id = event.get('payload', {}).get('target')
            # Mapping from 'L1_Buffer_A' to 'l1_buffer_a' happens in patterns.py, 
            # so payload['target'] should already be the lowercase ID.
            if dest_id and station.active_pallet_id:
                self.inventory[dest_id].expected_pallet_id = station.active_pallet_id
                print(f"  >>> SHUTTLE PUSH: {station.active_pallet_id} -> {dest_id}")

        # 2. LINEAR (Buffer/Dispenser): Use the Graph Topology
        elif event.get('type') == "STOPPER_OPEN":
            nodes = getattr(station, 'downstream_nodes', [])
            if len(nodes) == 1 and station.active_pallet_id:
                next_id = nodes[0]
                self.inventory[next_id].expected_pallet_id = station.active_pallet_id
                print(f"  >>> LINEAR PUSH: {station.active_pallet_id} -> {next_id}")
            else:
                print(f"  >>> LINEAR PUSH FAILED: Nodes={nodes}, HasID={bool(station.active_pallet_id)}")


    def _apply_inference(self, station, inference_dict):
        """Forces the virtual sensors to match the inferred reality."""
        for key, value in inference_dict.items():
            self.virtual_state_map[station.id][key] = value

    def _generate_metrics(self, station, state_def, timestamp):
        """
        Calculates values based on Config and adds to Buffer.
        """
        metrics_config = state_def.get('metrics', [])
        if not metrics_config: return

        for m_conf in metrics_config:
            m_name = m_conf['name']
            m_type = m_conf['type']
            value = None

            if m_type == 'duration_seconds':
                # Calculate Duration (Current Time - Entry Time)
                if station.state_entry_time > 0:
                    value = timestamp - station.state_entry_time
            
            # Future types: 'snapshot_value' (from payload), 'counter'

            if value is not None:
                # Add to Pre-Allocated Buffer
                self._add_to_buffer(timestamp, station.id, m_name, value, station.current_state)

    def _add_to_buffer(self, timestamp, comp_id, name, value, context):
        """
        O(1) Append. Flushes when full.
        """
        # Format Timestamp for readability in CSV (optional, can be raw epoch)
        ts_iso = datetime.fromtimestamp(timestamp).isoformat()
        
        row = (ts_iso, comp_id, name, round(value, 4), "s", context)
        self.buffer[self.buf_idx] = row
        self.buf_idx += 1
        
        if self.buf_idx >= self.buffer_size:
            self.flush_buffer()

    def flush_buffer(self):
        """Batch write to disk."""
        if self.buf_idx == 0: return

        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write slice [0 : idx]
            writer.writerows(self.buffer[:self.buf_idx])
            
        self.buf_idx = 0