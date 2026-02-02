Here is the **Updated Low-Level Design Document (v1.1)**. I have explicitly expanded Section 6 (Analytics) and added Section 9 (Future Proofing) to formalize how this architecture will evolve from a "Log Parser" to a "Physics-AI Hybrid Twin."

---

# **Low-Level Design Document (LLD)**

**Project:** Predictive Digital Twin for Glue/Primer Dispensing Unit
**Version:** 1.1 (Updated for Sensor/AI Extensibility)
**Architecture:** Event-Sourced, Configuration-Driven Graph

---

## **1. System Overview**

The system is an extensible **Digital Twin** that mirrors the physical state of a manufacturing unit.

* **Current Phase:** A "Process Twin" driven by log parsing to track flow, cycle times, and logic errors.
* **Future Phase:** A "Physics/AI Twin" driven by sensor fusion (Logs + IoT) to predict mechanical failure and quality issues.

**Core Philosophy:** The "Twin" is the single source of truth. It decouples **Input** (Logs/Sensors) from **Logic** (State Machine) and **Intelligence** (Rules/AI).

---

## **2. Architectural Patterns**

### **A. Graph Topology (The Physical Model)**

The machine is modeled as a **Directed Graph**, not a strict tree.

* **Nodes:** Stations (Shuttle, Dispenser) and Devices (Lifter, Stopper).
* **Edges:** Logical flow connecting stations (defined via `next_nodes` in config).
* **Shared Resources:** Nodes (e.g., Inspection Gantry) referenced by multiple upstream lines.

### **B. Unified Event Bus (The Nervous System)**

The system is **Input Agnostic**. It does not care *where* data comes from, only that an event occurred.

* **Log Input:** `LogParser`  emits `TwinEvent(type="STATE_CHANGE")`
* **Sensor Input (Future):** `MQTTClient`  emits `TwinEvent(type="TELEMETRY", payload={amps: 12.5})`

### **C. Strategy Pattern (The Pluggable Brain)**

Intelligence is separated from the machine structure. We can swap "Simple Rules" for "Physics Models" without rewriting the machine class.

---

## **3. Component Design**

### **3.1. Class Hierarchy**

* **`BaseComponent` (Abstract)**
* *Attributes:* `id`, `state` (IDLE, BUSY, ERROR), `config_params`.
* *Methods:* `handle_event(event)`, `attach_strategy(strategy)`.


* **`ShuttleStation`**: Handles 1-to-N routing and "Retry" logic.
* **`SharedStation`**: Handles Job Queuing (FIFO/Priority) for the Inspection Gantry.
* **`DispenseStation`**: Manages the critical path (Stopper  Lift  Dispense).

### **3.2. The Factory (Builder)**

* **`MachineBuilder`**: Reads `machine_config.yaml`, instantiates classes via Registry, and links the graph nodes.

---

## **4. Data Model**

### **4.1. The Pallet (The Product)**

```json
{
  "id": "A123",
  "current_node": "dispenser_l1",
  "history": [
    {"station": "shuttle", "dwell_time": 5.2, "retry_count": 0}
  ]
}

```

### **4.2. The TwinEvent (The Signal)**

```python
class TwinEvent:
    timestamp: float
    source_id: str      # e.g., "dispenser_l1"
    event_type: str     # "LOG_PATTERN" or "SENSOR_READING"
    payload: dict       # {"pattern": "Scan Fail"} OR {"vibration": 4.2g}

```

---

## **5. Configuration Schema**

**`machine_config.yaml`**

```yaml
components:
  - id: "dispenser_l1"
    class: "DispenserStation"
    # The brain is configurable here
    analytics_strategy: "StatisticalBaseline" # Can change to "PhysicsModel" later
    params:
       baseline_cycle_time: 15.0
       max_vibration_threshold: 2.5 # Future use
    log_patterns:
       "Servo Start": "DISPENSE_START"

```

---

## **6. Analytics Engine (The Extensible Brain)**

We define a standard interface `AnalyticsStrategy` with a `process(context, data)` method.

| Strategy Type | Logic Source | Use Case |
| --- | --- | --- |
| **`RuleBasedStrategy`** | `if time > limit: alert` | Cycle time monitoring, Stalled pallets. |
| **`StatisticalStrategy`** | `if val > mean + 3*sigma` | Detecting slow degradation (e.g., conveyor belt slip). |
| **`PhysicsStrategy`** |  | **Future:** Calculating work done by the lifter to detect friction/jamming. |
| **`AIInferenceStrategy`** | `model.predict(last_50_events)` | **Future:** Predicting glue clogging based on flow-rate + pressure sensors. |

---

## **7. Visualization (Observer Pattern)**

* **Subject:** `DigitalTwin` broadcasts state changes.
* **Observers:**
1. `ConsoleDashboard`: Text-based real-time table.
2. `JsonLRecorder`: Appends events to file for history.
3. `Visualizer3D (Future)`: Listens to socket for `(x,y,z)` updates.



---

## **8. Future Proofing & Sensor Integration**

This section outlines how we upgrade the system without refactoring.

### **Phase 1: Log-Only (Current)**

* **Goal:** Flow optimization.
* **Input:** Text Logs.
* **Logic:** "If 'Scan Fail' log appears > 3 times, warn User."

### **Phase 2: Hybrid (Logs + Sensors)**

* **Goal:** Predictive Maintenance.
* **Action:** Add `MQTTSensorIngestor` class.
* **Fusion Logic:**
* The Log tells us **Context**: *"The machine is currently dispensing."*
* The Sensor tells us **Reality**: *"The current draw is 15 Amps."*
* **Rule:** If `State == DISPENSING` AND `Current > 12A`, then `Alert: "High Viscosity/Clog"`. (We ignore high current during acceleration phases).



### **Phase 3: AI-Driven**

* **Goal:** Anomaly Detection.
* **Action:** Train an Isolation Forest or LSTM model on the `JsonLRecorder` history data.
* **Deployment:** Create `AIInferenceStrategy` that loads the `.onnx` model and scores events in real-time.

---

## **9. Next Step**

With the design locked and future-proofed, we can start the implementation.

**Action:** Please provide the **first snippet of logs** (e.g., the Shuttle/Scanner interaction). I will generate:

1. The `machine_config.yaml` for that section.
2. The `LogParser` regex.
3. The `ShuttleStation` Python class.