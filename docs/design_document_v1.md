This Low-Level Design (LLD) document serves as the blueprint for the **Predictive Rule-Based Digital Twin**. It is designed to be modular, configuration-driven, and scalable to accommodate future machine layouts.

---

# **Low-Level Design Document (LLD)**

**Project:** Digital Twin for Glue/Primer Dispensing Unit
**Version:** 1.0
**Architecture Type:** Event-Sourced, Configuration-Driven Graph

---

## **1. System Overview**

The system is a "Process Twin" that mirrors the physical state of a manufacturing unit by parsing real-time logs. It is designed to decouple the **topology** of the machine (defined in configuration) from the **behavior** of the components (defined in code).

**Core Goals:**

1. **Real-time Monitoring:** Tracking the location and state of every pallet.
2. **Performance Analytics:** Calculating cycle times (Record Mode) and detecting anomalies (Monitor Mode).
3. **Process Diagnostics:** Identifying bottlenecks or specific failure modes (e.g., Shuttle Retry fatigue).

---

## **2. Architectural Patterns**

### **A. Graph Topology (The Physical Model)**

The machine is not a hierarchy but a **Directed Graph**.

* **Nodes:** Stations (Shuttle, Dispenser, Buffers) and Devices (Lifter, Stopper).
* **Edges:** Conveyor logic connecting stations (defined via `next_nodes` in config).
* **Shared Resources:** Special nodes (e.g., Inspection Gantry) that multiple "Lines" point to, requiring queue/prioritization logic.

### **B. Event-Driven State Changes**

The system is reactive. It does not poll for status; it waits for events.

* `Log Line`  `Parser`  `TwinEvent`  `Component`  `State Update`

### **C. Observer Pattern (Visualization)**

To allow future 2D/3D visualization without refactoring core logic, the Twin acts as a `Subject`.

* The Twin broadcasts events (e.g., `PALLET_MOVED`).
* Subscribers (ConsoleLogger, JSONWriter, UnitySocket) listen and render independently.

---

## **3. Component Design**

### **3.1. Class Hierarchy**

* **`BaseComponent` (Abstract)**
* *Attributes:* `id`, `state` (IDLE, BUSY, ERROR), `config_params` (dict), `connected_nodes` (list).
* *Methods:* `handle_event(event)`, `get_snapshot()`, `reset()`.
* *Role:* The standardized interface for all machine parts.


* **`ShuttleStation` (Inherits BaseComponent)**
* *Specific Logic:* Handles complex routing (1-to-N). Manages "Retry Counters" for the scanner.
* *State Variables:* `retry_count`, `current_verification_mode`.


* **`LineStation` (Inherits BaseComponent)**
* *Specific Logic:* Standard "Stopper -> Lifter -> Process -> Exit" flow.
* *Sub-components:* Can own "Device" objects internally (e.g., `Lifter`, `Servo`) if granular logging exists.


* **`SharedStation` (Inherits BaseComponent)**
* *Specific Logic:* Implements a `JobQueue`. It accepts requests from multiple upstream nodes (Line 1 & Line 2) and processes them (FIFO or Priority).



### **3.2. The Factory (Builder)**

* **`MachineBuilder` Class:**
* Reads `machine_config.yaml`.
* Dynamically instantiates the correct Python classes based on string names.
* Resolves the graph links (converts string IDs `["gantry"]` to object references).



---

## **4. Data Model**

### **4.1. The Pallet (Transient Entity)**

Represents the product moving through the graph.

```json
{
  "id": "UUID_OR_BARCODE",
  "entry_timestamp": "2023-10-27T10:00:00",
  "current_node": "dispenser_l1",
  "history": [
    {"station": "shuttle", "dwell_time": 5.2, "status": "OK"},
    {"station": "buffer_a", "dwell_time": 12.0, "status": "WAIT"}
  ]
}

```

### **4.2. The Event (Internal Signal)**

Normalized object created by the Log Parser.

```python
class TwinEvent:
    timestamp: float
    target_component_id: str  # e.g., "shuttle_scanner"
    event_type: str           # e.g., "SCAN_FAIL"
    payload: Any              # e.g., { "retry_attempt": 3 }

```

---

## **5. Configuration Schema**

This file is the single source of truth for the machine's layout and log parsing logic.

**`machine_config.yaml` Draft:**

```yaml
machine:
  name: "DualLine_Glue_Dispenser"

components:
  # --- The Shuttle ---
  - id: "shuttle_main"
    class: "ShuttleStation"
    next_nodes: ["buffer_l1", "buffer_l2", "buffer_wait"]
    log_patterns:
      "Barcode read fail count (\d+)": "SCAN_RETRY"
      "Routing to Line 1": "ROUTE_L1"

  # --- Line 1 ---
  - id: "buffer_l1"
    class: "BufferStation"
    next_nodes: ["dispenser_l1"]
    
  - id: "dispenser_l1"
    class: "DispenserStation"
    next_nodes: ["inspection_gantry"] # Pointing to shared resource
    params:
        cycle_time_alert: 15.0 # Seconds

  # --- Shared Resource ---
  - id: "inspection_gantry"
    class: "SharedStation"
    next_nodes: ["exit_point"]
    params:
        source_priority: "FIFO"

```

---

## **6. Analytics Engine (Strategy Pattern)**

To switch between "Baselining" (Record Mode) and "Predictive Warnings" (Monitor Mode) dynamically:

* **`AnalyticsContext`**: Holds the logic strategy.
* **`RecordStrategy`**:
* On `TaskComplete`: Stores duration in a list `[4.1, 4.3, 4.2]`.
* Compute: Updates `mean` and `std_dev` constantly.


* **`MonitorStrategy`**:
* On `TaskComplete`: Checks `if duration > (mean + 3*std_dev)`.
* *Predictive Rule:* If "Retry Count" > 3 in last 5 pallets  Trigger "Maintenance Warning" (Lens cleaning required).



---

## **7. Visualization & Debugging**

### **7.1. Console Dashboard (Phase 1)**

A refreshable text view.

```text
[SHUTTLE]       [LINE 1]            [LINE 2]            [INSPECTION]
State: IDLE     State: BUSY         State: IDLE         State: BUSY (L1)
Retries: 0      Pallet: #8839       Pallet: --          Pallet: #8839
                Step: Dispensing                        Step: Imaging

```

### **7.2. Data Export (Phase 2)**

* **`ReplayLog`**: A JSONL file containing the entire state history. This can be fed into a future 3D tool (Unity/Three.js) to "replay" the day's production issues.

---

## **8. Failure Handling & Edge Cases**

* **Log Desynchronization:** If a "Start" log is found but "End" is missing (system crash), the component triggers a `STALE_STATE` timeout after  seconds.
* **Unknown Logs:** The parser logs unmatched lines to `unparsed.log` to help refine regex patterns later.
* **Shared Resource Collision:** If both Line 1 and Line 2 try to enter the Inspection Gantry simultaneously in the logs (rare but possible due to timestamp granularity), the `SharedStation` logic forces a serial order based on log arrival.

---

### **Next Step**

Does this design document cover the scope accurately?

If **Approved**, I will generate the **Python Code** following this exact structure:

1. `config.yaml`
2. `domain_classes.py` (Components)
3. `engine.py` (The Log Parser & Twin Controller)
4. `main.py` (Entry point)