"""
graph.py
--------
Two-node LangGraph pipeline:
  1. summarize     — runs all query functions for a given day, calls LLM (twice)
  2. write_summary — writes action_items + machine_health to Postgres
"""

import json
import requests
import psycopg2
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

import queries


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    run_date:       str
    start:          str
    end:            str
    action_items:   str   # ← new
    machine_health: str   # ← new


# ─────────────────────────────────────────────
# CONTEXT BUDGET
# ─────────────────────────────────────────────

MAX_CONTEXT_TOKENS = 4096
RESERVED_OUTPUT    = 700
SAFETY_BUFFER      = 50


# ─────────────────────────────────────────────
# POSTGRES HELPER
# ─────────────────────────────────────────────

def _get_connection(pg_config: dict):
    return psycopg2.connect(**pg_config)


# ─────────────────────────────────────────────
# TOKENIZER HELPER
# ─────────────────────────────────────────────

def count_tokens(text: str, vllm_base_url: str, model: str) -> int:
    try:
        base = vllm_base_url.rstrip("/").removesuffix("/v1")
        resp = requests.post(
            f"{base}/tokenize",
            json={"model": model, "prompt": text},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["count"]
    except Exception as e:
        print(f"  ⚠ Tokenizer call failed ({e}), falling back to char estimate")
        return int(len(text) / 2.5)


# ─────────────────────────────────────────────
# DATA CONDENSER
# ─────────────────────────────────────────────

def condense_rows(label: str, rows: list, max_rows: int = 10) -> str:
    if not rows:
        return f"[{label}]: No data."
    sample = rows[:max_rows]
    lines = [", ".join(f"{k}={v}" for k, v in row.items()) for row in sample]
    suffix = f" (+{len(rows) - max_rows} more)" if len(rows) > max_rows else ""
    return f"[{label} — {len(rows)} rows{suffix}]\n" + "\n".join(lines)


# ─────────────────────────────────────────────
# SHARED: trim data to token budget
# ─────────────────────────────────────────────

def _trim_to_budget(combined: str, header: str, footer: str,
                    vllm_base_url: str, vllm_model: str) -> tuple[str, int]:
    """Returns (trimmed_combined, safe_output_tokens)."""
    shell_tokens    = count_tokens(header + footer, vllm_base_url, vllm_model)
    max_data_tokens = MAX_CONTEXT_TOKENS - RESERVED_OUTPUT - SAFETY_BUFFER - shell_tokens
    combined_tokens = count_tokens(combined, vllm_base_url, vllm_model)

    print(f"  → shell tokens: {shell_tokens}, data tokens: {combined_tokens}, budget: {max_data_tokens}")

    if combined_tokens > max_data_tokens:
        ratio    = max_data_tokens / combined_tokens
        combined = combined[:int(len(combined) * ratio * 0.95)]
        combined += "\n[Data truncated]"
        print(f"  ⚠ Data truncated to {len(combined)} chars")

    prompt              = header + "\n" + combined + footer
    actual_tokens       = count_tokens(prompt, vllm_base_url, vllm_model)
    safe_max_tokens     = MAX_CONTEXT_TOKENS - actual_tokens - SAFETY_BUFFER
    safe_max_tokens     = max(100, min(safe_max_tokens, RESERVED_OUTPUT))
    print(f"  → exact prompt tokens: {actual_tokens}, output budget: {safe_max_tokens}")

    return combined, safe_max_tokens


# ─────────────────────────────────────────────
# NODE FACTORIES
# ─────────────────────────────────────────────

def make_summarize_node(pg_config: dict, llm: ChatOpenAI, vllm_base_url: str, vllm_model: str):
    def summarize_node(state: AgentState) -> AgentState:
        start    = state["start"]
        end      = state["end"]
        run_date = state["run_date"]
        print(f"\n[node] summarize  date={run_date}")

        conn = _get_connection(pg_config)

        query_functions = [
            ("Alert Volume by Category",                    queries.get_alert_volume_by_category),
            ("3σ Anomaly Events — All Stations & Metrics",  queries.get_anomaly_events),
            ("Top 20 Most Anomalous Metrics",               queries.get_top_anomalous_metrics),
            ("Process Groupwise Distribution of Anomalies", queries.get_anomaly_by_station),
            ("Error Logs",                                  queries.get_error_logs),
            ("Warning Logs",                                queries.get_warning_logs),
            ("Maintenance Events Over Time",                queries.get_maintenance_volume),
        ]

        sections: list[str] = []
        for label, fn in query_functions:
            try:
                rows = fn(conn, start, end)
                print(f"  → {label}: {len(rows)} rows")
                sections.append(condense_rows(label, rows))
            except Exception as e:
                print(f"  [ERROR] {label}: {e}")
                sections.append(f"[{label}]: Error — {e}")

        conn.close()

        combined = "\n\n".join(sections)

        # ── PROMPT 1: Action Items ──────────────────────────────────────────
        print("\n  [LLM 1/2] Generating Action Items …")

        action_header = f"""You are writing a daily machine health report for an operations engineer on the shop floor. They are not a data scientist — write in plain, simple English.

Based on the data below, produce ONLY the Action Items section for {run_date}.

IMPORTANT: Your entire response must be valid Markdown.
IMPORTANT: Total bullets across all three sections must not exceed 5. Aim for 1-2 bullets per sub-section.

Format exactly as:

## ✅ Action Items\n

**Do Now:**\n
- ...\n

---\n

**Do This Week:**\n
- ...\n

---\n

**Keep an Eye On:**\n
- ...\n

---\n

Rules:
- Write like you are telling a colleague what to do — short, direct, plain English.
- DO NOT just describe the problem — always say what action to take (e.g. "Inspect the front barrier actuator at Station 2 — it failed to extend 22 times today").
- Prioritise by severity: faults that happened more than 10 times go under "Do Now", recurring warnings go under "Do This Week", single-occurrence oddities go under "Keep an Eye On".
- Name the exact part, sensor, or station from the data — never say "the system" or "the machine" generically.
- Each bullet is ONE sentence, max 15 words.
- No sub-bullets, no extra headings, no preamble.
- Do not include any text before `##  Action Items` or after the last `---`.

DATA:
"""
        action_footer = "\n\nWrite the Action Items section now:"

        trimmed_combined, safe_tokens = _trim_to_budget(
            combined, action_header, action_footer, vllm_base_url, vllm_model
        )
        action_prompt = action_header + "\n" + trimmed_combined + action_footer

        action_response = llm.invoke(
            [HumanMessage(content=action_prompt)],
            max_tokens=safe_tokens,
        )
        action_items = action_response.content
        print(f"  → Action Items generated ({len(action_items)} chars)")

        # ── PROMPT 2: Machine Health ────────────────────────────────────────
        print("\n  [LLM 2/2] Generating Machine Health …")

        health_header = f"""You write a daily machine health report for an operations engineer on the shop floor. They are not a data scientist. Use plain English. Name specific parts and stations — never say "the system" alone.

Today's date: {run_date}

---

DATA DEFINITIONS (use these to interpret field names):
- Alerts = threshold breaches on sensors (temperature, pressure, flow rate). Tell the operator WHAT to watch and WHERE.
- Anomalies = statistical deviations detected by the anomaly model. Report COUNT only, not the same number as alerts.
- Errors = logged fault events with an exact error_name field. Report name + count + component.
- Maintenance events = IGNORE entirely. Do not mention them anywhere.

STATUS THRESHOLDS:
- Good: 0 errors, ≤2 alerts, ≤5 anomalies
- Warning: 1–2 errors OR 3–6 alerts OR 6–15 anomalies
- Critical: 3+ errors OR 7+ alerts OR 16+ anomalies
If multiple thresholds apply, use the worst one.

---

OUTPUT FORMAT — produce exactly these 4 sections, nothing before ## and nothing after the last ---:

## Overall Health

**Status:** [Good / Warning / Critical] — one sentence: the single biggest reason for this status, naming the specific part or station.

---

## ⚠️ Alerts

One sentence: what the alerts are telling the operator to watch out for and which part needs attention. Do NOT list counts — explain the risk.

---

## What Broke

One sentence: the exact error_name, how many times it occurred, and the specific component or station affected.
If no errors: write "No faults logged today."

---

## Unusual Behaviour

One sentence: total anomaly count, the worst station by name, and the single most anomalous sensor with its count.
If no anomalies: write "No unusual behaviour detected today."

---

HARD RULES (follow all, always):
1. Every section body = exactly 1 sentence. No bullets, no sub-headings.
2. If a metric name must appear, add a plain-English label in brackets: e.g. main_valve_temp (valve temperature).
3. Include real numbers from the data — never vague language like "several" or "some".
4. Never mention maintenance, maintenance alerts, or maintenance counts anywhere.
5. Never use "system" alone — always name the specific part, station, or component.
6. Alerts ≠ Anomalies — use the correct number for each section.
7. No text before ## Overall Health.
8. No text after the final ---.

DATA:
"""
        health_footer = "\n\nWrite the Machine Health sections now:"

        trimmed_combined, safe_tokens = _trim_to_budget(
            combined, health_header, health_footer, vllm_base_url, vllm_model
        )
        health_prompt = health_header + "\n" + trimmed_combined + health_footer

        health_response = llm.invoke(
            [HumanMessage(content=health_prompt)],
            max_tokens=safe_tokens,
        )
        machine_health = health_response.content
        print(f"  → Machine Health generated ({len(machine_health)} chars)")

        return {
            **state,
            "action_items":   action_items,
            "machine_health": machine_health,
        }

    return summarize_node


def make_write_summary_node(pg_config: dict, summary_table: str):
    def write_summary_node(state: AgentState) -> AgentState:
        print(f"[node] write_summary  date={state['run_date']}")
        try:
            conn = _get_connection(pg_config)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO "{summary_table}" (run_date, action_items, machine_health)
                    VALUES (%s, %s, %s);
                    """,
                    (
                        state["run_date"],
                        state.get("action_items",   ""),
                        state.get("machine_health", ""),
                    ),
                )
            conn.commit()
            conn.close()
            print(f"  → written to `{summary_table}` (action_items + machine_health)")
        except Exception as e:
            print(f"  [ERROR] write_summary failed: {e}")

        return state

    return write_summary_node


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────

def build_graph(
    pg_config:     dict,
    llm:           ChatOpenAI,
    summary_table: str = "agent_summaries",
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    vllm_model:    str = "Qwen/Qwen3-14B-AWQ",
):
    graph = StateGraph(AgentState)

    graph.add_node("summarize",     make_summarize_node(pg_config, llm, vllm_base_url, vllm_model))
    graph.add_node("write_summary", make_write_summary_node(pg_config, summary_table))

    graph.set_entry_point("summarize")
    graph.add_edge("summarize",     "write_summary")
    graph.add_edge("write_summary", END)

    return graph.compile()