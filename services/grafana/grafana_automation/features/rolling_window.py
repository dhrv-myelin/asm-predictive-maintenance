"""
Rolling Window Feature
-----------------------
Handles rolling window configuration for the dashboard time range.

When enabled, overrides the dashboard's default time range with a
rolling window (e.g. last 7 days always shown on load).

To enable: set features.rolling_window.enabled to true in config
and set the desired window (e.g. '7d', '24h', '30d').
"""

# Mapping of human-friendly window strings to Grafana 'from' values
WINDOW_MAP = {
    "1h":  "now-1h",
    "3h":  "now-3h",
    "6h":  "now-6h",
    "12h": "now-12h",
    "24h": "now-24h",
    "2d":  "now-2d",
    "7d":  "now-7d",
    "14d": "now-14d",
    "30d": "now-30d",
    "90d": "now-90d",
}


def build(config: dict) -> dict:
    """
    Build the time range dict for the dashboard.

    If rolling_window is disabled, returns None (caller keeps existing time range).
    If enabled, returns a Grafana time dict with 'from' and 'to'.

    Args:
        config: Full parsed config dict

    Returns:
        Dict with 'from' and 'to' keys, or None if feature is disabled
    """
    rolling_cfg = config.get("features", {}).get("rolling_window", {})

    if not rolling_cfg.get("enabled", False):
        return None

    window = rolling_cfg.get("window", "7d")
    from_value = WINDOW_MAP.get(window)

    if from_value is None:
        raise ValueError(
            f"Unknown rolling window value: '{window}'. "
            f"Supported values: {list(WINDOW_MAP.keys())}"
        )

    return {
        "from": from_value,
        "to": "now",
    }
