"""
src/tag_resolver.py
Maps hardware tags (e.g., "L1BufferAPalletPresenceSensorDI") back to Station IDs (e.g., "l1_buffer_a").
"""
import yaml
import os

class TagResolver:
    def __init__(self, graph_path):
        self.tag_map = self._build_reverse_map(graph_path)

    def _build_reverse_map(self, path):
        mapping = {}
        if not os.path.exists(path):
            print(f"⚠️ TagResolver: Config file not found at {path}")
            return mapping

        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                
            for node in config.get('topology', []):
                station_id = node['id']
                hw = node.get('hardware_link', {})
                
                # Index Sensors
                sensors = hw.get('sensors', {})
                for _, tag_name in sensors.items():
                    mapping[tag_name] = station_id
                    
                # Index Actuators
                actuators = hw.get('actuators', {})
                for _, tag_name in actuators.items():
                    mapping[tag_name] = station_id
                    
        except Exception as e:
            print(f"⚠️ TagResolver Error: {e}")
            
        return mapping

    def resolve(self, tag_name):
        """Returns station_id or None"""
        return self.tag_map.get(tag_name)
    