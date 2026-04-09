"""
Annotations Feature
--------------------
Builds the Grafana annotations list for the dashboard.

Currently includes the default Grafana built-in annotation.
When nelson rules are ready, each rule defined in config under
features.annotations.rules gets added as a separate annotation layer.

To add a new Nelson rule later:
  1. Add the SQL query target to templates/spc_timeseries.py (new refId)
  2. Add the rule definition to config yaml under features.annotations.rules
  3. Nothing else changes here — this builder loops over whatever rules exist
"""


def _default_annotation() -> dict:
    """Grafana built-in annotation — always present."""
    return {
        "builtIn": 1,
        "datasource": {"type": "grafana", "uid": "-- Grafana --"},
        "enable": True,
        "hide": True,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard",
    }


def _nelson_rule_annotation(rule: dict, datasource: dict, station_name: str) -> dict:
    """
    Build a single Nelson rule annotation entry.

    Args:
        rule:         Dict from config with 'name', 'color', 'query' keys
        datasource:   Datasource dict with 'type' and 'uid'
        station_name: Station name for SQL filtering
    """
    return {
        "datasource": datasource,
        "enable": True,
        "hide": False,
        "iconColor": rule.get("color", "red"),
        "name": rule["name"],
        "rawQuery": True,
        "rawSql": _build_nelson_sql(station_name, rule["query"]),
        "showIn": 0,
        "step": "60s",
        "type": "alert",
        "useValueForTime": False,
    }


def _build_nelson_sql(station_name: str, rule_query: str) -> str:
    """
    Placeholder SQL for Nelson rule annotations.
    Replace with actual rule SQL when implementing each rule.

    Args:
        station_name: Station to filter on
        rule_query:   Rule identifier from config (e.g. 'nelson_rule_1')
    """
    return (
        "SELECT\n"
        "  timestamp AS time,\n"
        f"  '{rule_query}' AS text,\n"
        f"  '{rule_query}' AS tags\n"
        "FROM nelson_violations\n"
        "WHERE $__timeFilter(timestamp)\n"
        f"  AND station_name = '{station_name}'\n"
        f"  AND rule_name = '{rule_query}'\n"
        "ORDER BY time"
    )


def build(config: dict) -> list:
    """
    Build the full annotations list for the dashboard.

    Always includes the Grafana built-in annotation.
    If features.annotations.enabled is true, also adds each nelson rule.

    Args:
        config: Full parsed config dict

    Returns:
        List of annotation dicts for dashboard['annotations']['list']
    """
    annotations = [_default_annotation()]

    annotations_cfg = config.get("features", {}).get("annotations", {})

    if not annotations_cfg.get("enabled", False):
        return annotations

    datasource = config["datasource"]
    station_name = config["station_name"]
    rules = annotations_cfg.get("rules", [])

    for rule in rules:
        annotations.append(
            _nelson_rule_annotation(rule, datasource, station_name)
        )

    return annotations
