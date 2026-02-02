"""
tools/generate_dummy_logs.py
Generates logs using REAL IO TAGS defined in machine_graph.yaml.
Restored logical density to prevent state jumps.
"""
import os
from datetime import datetime, timedelta

OUTPUT_FILE = "data/machine_logs.txt"
START_TIME = datetime(2026, 1, 21, 10, 0, 0)

# SCENARIO: 
# Using format: "IO: Sensor <TAG_ID> triggered"
# Logic events use specific descriptive strings matched by regex patterns.

SCENARIO = [
    (0.0, "System", "Machine initialized"),

    # ==========================================
    # LINE 1 FLOW
    # ==========================================
    
    # 1. Shuttle (Entry & Routing)
    (1.0,  "ShuttlePLC", "IO: Sensor ShuttlePalletPresenceSensorDI triggered"),
    (2.0,  "ShuttlePLC", "Pallet P_001 detected at Shuttle"), 
    (3.0,  "ShuttlePLC", "Pallet P_001 sent to L1_Buffer_A successfully"),
    (5.5,  "ShuttlePLC", "Shuttle Axis Arrived"),
    (6.5,  "ShuttlePLC", "Transfer to L1_Buffer_A complete"),

    # 2. Buffer A
    (8.0,  "BufferPLC",  "IO: Sensor L1BufferAPalletPresenceSensorDI triggered"),
    (10.0, "BufferPLC",  "L1_Buffer_A Release"), 
    (11.0, "BufferPLC",  "Pallet left L1_Buffer_A"),

    # 3. Dispenser L1
    (15.0, "DispenserL1", "IO: Sensor L1DispensingStationPalletPresenceSensorDI triggered"),
    (16.0, "DispenserL1", "Clamp Locked at L1_Dispenser"),
    (17.0, "DispenserL1", "Dispensing Started at L1_Dispenser"),
    (32.0, "DispenserL1", "Process finished at L1_Dispenser"),
    (33.0, "DispenserL1", "Clamp Unlocked at L1_Dispenser"),
    (34.0, "DispenserL1", "Dispenser L1_Dispenser Release"),
    (35.0, "DispenserL1", "Pallet left L1_Dispenser"),

    # 4. Buffer B
    (38.0, "BufferPLC",  "IO: Sensor L1BufferBPalletPresenceSensorDI triggered"),
    (40.0, "BufferPLC",  "L1_Buffer_B Release"),
    (41.0, "BufferPLC",  "Pallet left L1_Buffer_B"),

    # 5. Inspection (Shared)
    (45.0, "VisionSys",   "IO: Sensor L1InspectionStationPalletPresenceSensorDI triggered"),
    (46.0, "VisionSys",   "Inspection Start L1"),
    (50.0, "VisionSys",   "Inspection Result: PASS"),
    (51.0, "VisionSys",   "Pallet left Inspection Station"),


    # ==========================================
    # LINE 2 FLOW
    # ==========================================
    
    # 1. Shuttle (Entry & Routing)
    (20.0, "ShuttlePLC", "IO: Sensor ShuttlePalletPresenceSensorDI triggered"),
    (21.0, "ShuttlePLC", "Pallet P_002 detected at Shuttle"),
    (22.0, "ShuttlePLC", "Pallet P_002 sent to L2_Buffer_A successfully"),
    (24.5, "ShuttlePLC", "Shuttle Axis Arrived"),
    (25.5, "ShuttlePLC", "Transfer to L2_Buffer_A complete"),

    # 2. Buffer A
    (28.0, "BufferPLC",  "IO: Sensor L2BufferAPalletPresenceSensorDI triggered"),
    (29.0, "BufferPLC",  "L2_Buffer_A Release"),
    (30.0, "BufferPLC",  "Pallet left L2_Buffer_A"),

    # 3. Dispenser L2
    (35.0, "DispenserL2", "IO: Sensor L2DispensingStationPalletPresenceSensorDI triggered"),
    (36.0, "DispenserL2", "Clamp Locked at L2_Dispenser"),
    (37.0, "DispenserL2", "Dispensing Started at L2_Dispenser"),
    (55.0, "DispenserL2", "Process finished at L2_Dispenser"),
    (56.0, "DispenserL2", "Clamp Unlocked at L2_Dispenser"),
    (57.0, "DispenserL2", "Dispenser L2_Dispenser Release"),
    (58.0, "DispenserL2", "Pallet left L2_Dispenser"),
    
    # 4. Buffer B
    (61.0, "BufferPLC",  "IO: Sensor L2BufferBPalletPresenceSensorDI triggered"),
    (63.0, "BufferPLC",  "L2_Buffer_B Release"),
    (64.0, "BufferPLC",  "Pallet left L2_Buffer_B"),
    
    # 5. Inspection (Shared)
    (68.0, "VisionSys",   "IO: Sensor L2InspectionStationPalletPresenceSensorDI triggered"),
    (69.0, "VisionSys",   "Inspection Start L2"),
    (73.0, "VisionSys",   "Inspection Result: PASS"),
    (74.0, "VisionSys",   "Pallet left Inspection Station"),
]

def generate():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for offset_sec, service, msg in SCENARIO:
            ts = START_TIME + timedelta(seconds=offset_sec)
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3] 
            f.write(f"{ts_str} [MainThread] INFO  {service} - {msg}\n")
            
    print("Done. Logs generated with logical density and 'IO:' prefix.")

if __name__ == "__main__":
    generate()
