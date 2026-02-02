"""
src/viz_adapter.py
Visualizes the process state using Rerun.io
"""
import rerun as rr
import yaml
from datetime import datetime

class VizAdapter:
    def __init__(self, config_path, start_time_str=None):
        self.config = self._load_config(config_path)
        self.resources = self.config.get('resources', {})
        self.state_colors = self.config.get('state_colors', {})
        
        # Pallet Settings
        self.pallet_size = [0.4, 0.4, 0.1] 
        self.pallet_color = [255, 215, 0]

        # Time Anchoring (The Fix for 1970)
        # We try to parse the start string; if failing, we default to Now.
        try:
            # Assumes format "YYYY-MM-DD HH:MM:SS,mmm"
            # We strip the last 4 chars (,mmm) for simplicity or handle strictly
            if start_time_str:
                base = datetime.strptime(start_time_str.split(',')[0], "%Y-%m-%d %H:%M:%S")
                self.base_timestamp = base.timestamp()
            else:
                self.base_timestamp = datetime.now().timestamp()
        except Exception as e:
            print(f"Viz Warning: Could not parse start time '{start_time_str}'. Defaulting to Now.")
            self.base_timestamp = datetime.now().timestamp()

        rr.init("Industrial_Digital_Twin", spawn=True)
        self._setup_scene()

    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def _setup_scene(self):
        # Set timeline to the BASE time (2026...), not 0
        rr.set_time(timeline="stable_time", timestamp=self.base_timestamp)
        
        for entity_id, props in self.resources.items():
            pos = props.get('position', [0,0,0])
            size = props.get('size', [1,1,1])
            base_color = props.get('color', [100,100,100])
            
            rr.log(
                f"world/{entity_id}",
                rr.Boxes3D(
                    half_sizes=[s/2 for s in size], 
                    centers=[pos], 
                    labels=entity_id, 
                    colors=base_color
                )
            )

    def log_completion(self, absolute_time, total_count):
        """
        relative_time: Seconds since start (e.g., 15.0)
        """
        # Match the logic used in the 'update' method
        rr.set_time(timeline="stable_time", timestamp=absolute_time)
        
        # Log the scalar to a dedicated metrics path
        rr.log("metrics/throughput", rr.Scalars(total_count))


    def update_scoreboard(self, total_count):
        # Note: Scoreboards (TextDocuments) often don't need a timeline 
        # unless you want to see the history of the text change.
        summary = f"## 🏆 Production Summary\n\n**Total Completed:** {total_count}"
        rr.log("summary/stats", rr.TextDocument(summary, media_type="text/markdown"))


    def update(self, absolute_time, entity_id, state, context_info=None):
        """
        relative_time: Seconds since start (e.g., 15.0)
        """
        rr.set_time(timeline="stable_time", timestamp=absolute_time)
        
        if entity_id not in self.resources: return 

        props = self.resources[entity_id]
        pos = props.get('position')
        size = props.get('size')
        
        base_color = props.get('color', [100,100,100])
        new_color = self.state_colors.get(state, base_color)
        
        current_pallet = context_info.get('pallet_id') if context_info else None
        
        # Label
        pallet_str = f" | {current_pallet}" if current_pallet else ""
        label_text = f"{entity_id}\n[{state}]{pallet_str}"
        
        rr.log(
            f"world/{entity_id}",
            rr.Boxes3D(half_sizes=[s/2 for s in size], centers=[pos], colors=new_color, labels=label_text)
        )

        if current_pallet:
            machine_height = size[2]
            pallet_z_offset = (machine_height / 2) + (self.pallet_size[2] / 2)
            pallet_pos = [pos[0], pos[1], pos[2] + pallet_z_offset]

            rr.log(
                f"world/pallets/{current_pallet}",
                rr.Boxes3D(
                    half_sizes=[s/2 for s in self.pallet_size],
                    centers=[pallet_pos],
                    colors=self.pallet_color,
                    labels=current_pallet
                )
            )