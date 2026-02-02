"""
src/viz_adapter.py
Visualizes the process state AND the physical product flow.
"""
import rerun as rr
import yaml

class VizAdapter:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.resources = self.config.get('resources', {})
        self.state_colors = self.config.get('state_colors', {})
        
        # Define Pallet Dimensions (Standard Tray)
        self.pallet_size = [0.4, 0.4, 0.1] 
        self.pallet_color = [255, 215, 0] # Gold Color for visibility

        rr.init("Industrial_Digital_Twin", spawn=True)
        self._setup_scene()

    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"CRITICAL: Viz Config not found at {path}")
            return {}

    def _setup_scene(self):
        rr.set_time(timeline="stable_time", timestamp=0)
        
        for entity_id, props in self.resources.items():
            pos = props.get('position', [0,0,0])
            size = props.get('size', [1,1,1])
            base_color = props.get('color', [100,100,100])
            
            # Log the Machine
            rr.log(
                f"world/{entity_id}",
                rr.Boxes3D(
                    half_sizes=[s/2 for s in size], 
                    centers=[pos], 
                    labels=entity_id, 
                    colors=base_color
                )
            )

    def update(self, timestamp, entity_id, state, context_info=None):
        """
        Updates machine state AND moves the pallet if one is present.
        """
        rr.set_time(timeline="stable_time", timestamp=timestamp)
        
        if entity_id not in self.resources: return 

        # 1. Update Machine Visuals
        props = self.resources[entity_id]
        pos = props.get('position')
        size = props.get('size')
        
        base_color = props.get('color', [100,100,100])
        new_color = self.state_colors.get(state, base_color)
        
        # Check if a pallet is here
        pallet_id = context_info.get('pallet_id') if context_info else None
        
        # Label: Machine + Pallet ID
        label_text = f"{entity_id}\n[{state}]"
        
        rr.log(
            f"world/{entity_id}",
            rr.Boxes3D(
                half_sizes=[s/2 for s in size], 
                centers=[pos], 
                colors=new_color,
                labels=label_text
            )
        )

        # 2. Update Pallet Visuals (The "Pop" Logic)
        if pallet_id:
            # Calculate Pallet Position: Center of Machine + Z Offset (Height of machine/2 + Pallet Height/2)
            machine_height = size[2]
            pallet_z_offset = (machine_height / 2) + (self.pallet_size[2] / 2)
            
            pallet_pos = [pos[0], pos[1], pos[2] + pallet_z_offset]

            # Log the Pallet as a separate entity
            # Because we use the same ID "world/pallets/{pallet_id}", 
            # Rerun automatically "moves" it from the old location to this new one!
            rr.log(
                f"world/pallets/{pallet_id}",
                rr.Boxes3D(
                    half_sizes=[s/2 for s in self.pallet_size],
                    centers=[pallet_pos],
                    colors=self.pallet_color,
                    labels=pallet_id
                )
            )
        
        # OPTIONAL: If state is IDLE/EMPTY, we technically assume the pallet is gone.
        # But in a discrete log system, we don't know WHERE it went (conveyor).
        # So we just leave it at the last known station until it arrives at the next one.