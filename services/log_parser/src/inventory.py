"""
src/inventory.py
Defines the physical entities of the Digital Twin.
Acts as the 'Dumb Container' for state.
"""

class Station:
    def __init__(self, id, config, logic_template, io_map=None, axis_map=None):
        self.id = id
        self.config = config         # From machine_graph.yaml
        self.logic_template = logic_template  # From process_logic.yaml
        self.io_map = io_map if io_map else {}
        self.axis_map = axis_map if axis_map else {}

        self.is_entry = config.get('is_entry', False)
        self.is_exit = config.get('is_exit', False)
        
        # State Management
        self.current_state = logic_template.get('initial_state', 'IDLE')
        self.previous_state = None
        self.state_entry_time = 0.0  # Timestamp of last transition
        self.current_destination = None
        
        # Context (What is effectively "on" this station?)
        self.active_pallet_id = None
        self.expected_pallet_id = None
        
        # Metrics Buffer (Holds partial data before flush)
        # Structure: {'cycle_time': start_timestamp}
        self.metric_timers = {} 

    def set_state(self, new_state, timestamp):
        """
        Updates state and timestamps.
        Does NOT decide logic (Engine does that).
        """
        if new_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_entry_time = timestamp
            
        # If we return to IDLE or EMPTY, the pallet has physically left
        if new_state in ["IDLE", "EMPTY"]:
            self.active_pallet_id = None
        
        # Reset metric timers that are state-specific if needed
        # (Logic handled in Engine usually, but good to know state changed)

    def get_hardware_keys(self):
        """Helper to get the raw hardware map for this station"""
        return self.config.get('hardware_link', {})

    @property
    def downstream_nodes(self):
        """
        Returns the list of valid next stations from 'next_nodes' in machine_graph.yaml.
        Example: ["l1_buffer_b"] or ["l1_buffer_a", "l2_buffer_a"]
        """
        return self.config.get('next_nodes', [])


class SystemInventory:
    """
    The Factory that builds the World.
    """
    @staticmethod
    def build(graph_config, logic_config, io_config_full=None, gantry_config_full=None):
        inventory = {}

        # Default to empty dicts if files were missing
        io_config_full = io_config_full or {}
        gantry_config_full = gantry_config_full or {}
        
        for node in graph_config['topology']:
            station_id = node['id']
            logic_key = node.get('logic_template')
            
            # Validation: Ensure Logic exists for this Station
            if logic_key not in logic_config['definitions']:
                raise ValueError(f"Station '{station_id}' references missing logic template: '{logic_key}'")
                
            logic_def = logic_config['definitions'][logic_key]
            
            # Instantiate
            station = Station(station_id, node, logic_def)
            inventory[station_id] = station
            
        return inventory