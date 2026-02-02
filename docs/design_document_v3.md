# System Design: Industrial Log Analytics Engine

1. High-Level Architecture

The system is designed as a Linear Pipeline that transforms unstructured text logs into structured Time-Series Data (Metrics). It operates in two modes: Record (Baselining) and Monitor (Anomaly Detection).

2. Directory Structure

This standardizes the input/output locations to ensure versioning and cleanliness.
Plaintext

/project_root
├── /config                  # The Static Rules (Read-Only)
│   ├── machine_graph.yaml   # Topology & Hardware Keys
│   ├── process_logic.yaml   # State Machine, Inferences, Metric Defs
│   ├── log_patterns.yaml    # Regex Rules
│   └── viz_resources.yaml   # (Optional) 3D Attributes
│
├── /data                    # The Dynamic State (Read/Write)
│   ├── /baselines           # Archive of generated baselines
│   ├── /snapshots           # Archive of crash states
│   ├── machine_logs.txt     # SOURCE INPUT
│   ├── process_baseline.json# ACTIVE REFERENCE
│   ├── machine_state.json   # RESUME POINT
│   └── process_metrics.csv  # FINAL OUTPUT
│
└── /src                     # The Logic
    ├── domain_classes.py
    ├── log_parser.py
    ├── engine.py
    └── main.py

3. Module Responsibilities (The "Who Does What")
A. src/inventory.py (The State Containers)

    Component Class:

        Role: A "dumb" container. It knows who it is (ID) and what it is doing (State).

        Data Held: Current State, Previous State, Entry Timestamp, Active Pallet ID.

        NO Logic: It does not decide when to change state. It just stores the change.

    ComponentFactory:

        Role: Reads machine_graph.yaml and instantiates the Component objects.

B. src/log_parser.py (The Translator)

    Role: Converts raw text lines into Event dictionaries.

    Mechanism:

        Compiles Regex from log_patterns.yaml.

        Extracts Timestamps.

        Maps raw names (e.g., "L1") to IDs (l1_dispenser).

    Output: Yields a clean object: {'timestamp': 1001, 'type': 'PROCESS_START', 'target': 'l1_dispenser'}.

C. src/engine.py (The Brain)

    Role: The Controller. It holds the "Virtual Reality."

    Key Algorithms:

        State Inference: Before checking logic, it updates the virtual_sensor_map based on the event (solving the "Sparse Log" problem).

        Transition Logic: Checks if Current State + Event = Valid Transition.

        Metric Calculation: On State Exit, checks the Config to see if it needs to compute duration or capture a value.

        Buffer Management: Manages the pre-allocated list buffer and flushes to CSV.

D. src/main.py (The Driver)

    Role: Application Lifecycle Management.

    Flow:

        Boot: Load YAMLs -> Load machine_state.json (Warm Start).

        Loop: Feed Parser output into Engine.

        Shutdown: Save current state to machine_state.json -> Flush CSV.


4. Visualization Module (src/viz_adapter.py)

We will add a dedicated module for this. It acts as a Passive Observer.

    Role: It watches the Engine. When the Engine updates a state, the Adapter draws it.

    Why Separate? This ensures your core logic (Analytics) works even if you run it on a server with no screen. The Visualizer is just a "plugin."

Architecture:

    Input: Reads config/viz_resources.yaml (Colors/Shapes).

    Runtime: In engine.py, inside the process_event loop:

        After state change → Call viz_adapter.update(component_id, new_state).

    Technology: We use Rerun.io (Python SDK). It is perfect for this because it handles the "Time Slider" automatically. You just log data, and it builds the history.

Visual Debugging Features:

    State Colors: We can map states to colors (e.g., ERROR = Red, DISPENSING = Blue) in the adapter. This makes spotting anomalies instant.

5. The Data Lifecycle (Step-by-Step)

This is the exact path a single log line takes through the system.

Input: 2026-01-21 10:05:00 [INFO] Dispensing Started at L1

    Parsing (Regex):

        Matches pattern Dispensing Started at (?P<station>\w+).

        Maps L1 → l1_dispenser.

        Event: {type: "PROCESS_START", target: "l1_dispenser", time: 10:05:00}.

    Inference (Virtual Reality Update):

        Engine looks at process_logic.yaml for PROCESS_START.

        Does it have state_inference? (Maybe sensors.presence = True).

        Engine updates self.virtual_sensors['l1_dispenser']['presence'] = True.

    Logic Check:

        Current State: READY_TO_DISPENSE.

        Transition: PROCESS_START → DISPENSING.

        Hardware Check: (Passes because we just updated the virtual sensor).

    Metric Capture (The Output):

        We are leaving READY_TO_DISPENSE.

        Does this state have metrics? No.

        System moves to DISPENSING and records entry_time = 10:05:00.

    Later... (Next Log Line): 2026-01-21 10:05:15 [INFO] Process finished

        System moves DISPENSING → UNCLAMPING.

        Metric Trigger: leaving DISPENSING.

        Config says: name: process_cycle_time, type: duration.

        Calc: 10:05:15 - 10:05:00 = 15.0s.

        Buffer Append: [10:05:15, "l1_dispenser", "process_cycle_time", 15.0, "s"].

6. Handling Special Scenarios

    Scenario A: The Code Crashes.

        Defense: try...finally block in main.py.

        Action: Captures the in-memory Component objects and dumps them to data/snapshots/crash_state.json.

        Recovery: Next run loads this file to resume.

    Scenario B: The "Jump" (Permissive Mode).

        Context: Log misses "Clamping", goes straight to "Dispensing".

        Action: Engine detects invalid transition.

        Logic: It scans forward in the graph. If "Dispensing" is reachable, it auto-executes the intermediate steps (creating 0-second duration records for them) to catch up.

7. Output Schema (CSV)

We use the Tall Format as agreed.
timestamp	component_id	metric_name	value	unit	state_context
float/iso	string	string	float	str	string

    Partitioning: If the file grows > 100MB, we can simply start a new file (process_metrics_part2.csv) without breaking the schema.