"""
SPC Timeseries Panel Template
------------------------------
Generates a Grafana timeseries panel with:
  - Target A : raw value per-row + UCL / LCL / Mean + inline Anomaly column (SPC control lines)
  - Target B : rolling window Bollinger bands (rolling_mean, bb_upper, bb_lower)
               uses the ${rolling_window} Grafana variable for window size

All field overrides (colors, line styles, fill bands, rolling lines) match
the reference dashboard exactly.
"""

_SQL_STUB = {
    "columns": [{"parameters": [], "type": "function"}],
    "groupBy": [{"property": {"type": "string"}, "type": "groupBy"}],
    "limit": 50,
}


def _build_sql_main(station_name: str, metric_name: str) -> str:
    """Target A — per-row value + static UCL/LCL/Mean + inline Anomaly flag."""
    return (
        f"WITH filtered AS (\n"
        f"  SELECT timestamp, value\n"
        f"  FROM process_metrics\n"
        f"  WHERE $__timeFilter(timestamp)\n"
        f"    AND station_name = '{station_name}'\n"
        f"    AND metric_name = '{metric_name}'\n"
        f"),\n"
        f" \n"
        f"baseline AS (\n"
        f"  SELECT mean, std_dev\n"
        f"  FROM baseline_metrics\n"
        f"  WHERE station_name = '{station_name}'\n"
        f"    AND metric_name = '{metric_name}'\n"
        f"  LIMIT 1\n"
        f")\n"
        f" \n"
        f"SELECT\n"
        f"  f.timestamp AS time,\n"
        f"  f.value AS \"value\",\n"
        f" \n"
        f"  (b.mean + 3 * b.std_dev) AS \"UCL (+3\u03c3)\",\n"
        f"  (b.mean - 3 * b.std_dev) AS \"LCL (-3\u03c3)\",\n"
        f"  b.mean AS \"Mean\",\n"
        f" \n"
        f"  CASE\n"
        f"    WHEN f.value > (b.mean + 3 * b.std_dev)\n"
        f"      OR f.value < (b.mean - 3 * b.std_dev)\n"
        f"    THEN f.value\n"
        f"    ELSE NULL\n"
        f"  END AS \"Anomaly\"\n"
        f" \n"
        f"FROM filtered f\n"
        f"CROSS JOIN baseline b\n"
        f"ORDER BY f.timestamp;\n"
        f" "
    )


def _build_sql_rolling(station_name: str, metric_name: str) -> str:
    """Target B — rolling window Bollinger bands using ${rolling_window} Grafana variable."""
    return (
        f"WITH base AS (\n"
        f"  SELECT\n"
        f"    timestamp AS time,\n"
        f"    value\n"
        f"  FROM process_metrics\n"
        f"  WHERE $__timeFilter(timestamp)\n"
        f"    AND station_name = '{station_name}'\n"
        f"    AND metric_name = '{metric_name}'\n"
        f"),\n"
        f"rolling AS (\n"
        f"  SELECT\n"
        f"    time,\n"
        f"    AVG(value)         OVER w AS rolling_mean,\n"
        f"    STDDEV_SAMP(value) OVER w AS rolling_std\n"
        f"  FROM base\n"
        f"  WINDOW w AS (\n"
        f"    ORDER BY time\n"
        f"    RANGE BETWEEN ${{rolling_window:sqlstring}}::interval PRECEDING\n"
        f"    AND CURRENT ROW\n"
        f"  )\n"
        f")\n"
        f"SELECT\n"
        f"  time,\n"
        f"  rolling_mean,\n"
        f"  rolling_mean + 3 * rolling_std AS bb_upper,\n"
        f"  rolling_mean - 3 * rolling_std AS bb_lower\n"
        f"FROM rolling\n"
        f"ORDER BY time;"
    )


def _field_overrides(value_color: str, bb_upper_show_points: bool) -> list:
    """
    Field overrides. Two params vary per-panel based on config:
      value_color:           fixedColor for the 'value' series
      bb_upper_show_points:  whether to include custom.showPoints on bb_upper
    """
    bb_upper_props = [
        {"id": "color", "value": {"fixedColor": "dark-blue", "mode": "fixed"}},
        {"id": "custom.lineStyle", "value": {"dash": [10, 10], "fill": "dash"}},
        {"id": "displayName", "value": "Rolling Upper Bound (+3\u03c3)"},
    ]
    if bb_upper_show_points:
        bb_upper_props.append({"id": "custom.showPoints", "value": "never"})

    return [
        {
            "matcher": {"id": "byName", "options": "value"},
            "properties": [
                {"id": "color", "value": {"fixedColor": value_color, "mode": "fixed"}},
                {"id": "custom.lineWidth", "value": 3},
                {"id": "custom.showPoints", "value": "never"},
            ],
        },
        {
            "matcher": {"id": "byName", "options": "UCL (+3\u03c3)"},
            "properties": [
                {"id": "color", "value": {"fixedColor": "#1F60C4", "mode": "fixed"}},
                {"id": "custom.lineWidth", "value": 2},
                {"id": "custom.showPoints", "value": "never"},
                {"id": "custom.lineStyle", "value": {"fill": "solid"}},
                {"id": "custom.fillBelowTo", "value": "LCL (-3\u03c3)"},
                {"id": "custom.fillOpacity", "value": 6},
            ],
        },
        {
            "matcher": {"id": "byName", "options": "LCL (-3\u03c3)"},
            "properties": [
                {"id": "color", "value": {"fixedColor": "#1F60C4", "mode": "fixed"}},
                {"id": "custom.lineWidth", "value": 2},
                {"id": "custom.showPoints", "value": "never"},
                {"id": "custom.lineStyle", "value": {"fill": "solid"}},
            ],
        },
        {
            "matcher": {"id": "byName", "options": "Mean"},
            "properties": [
                {"id": "color", "value": {"fixedColor": "#FF9830", "mode": "fixed"}},
                {"id": "custom.lineWidth", "value": 2},
                {"id": "custom.showPoints", "value": "never"},
                {"id": "custom.lineStyle", "value": {"dash": [8, 4], "fill": "dash"}},
            ],
        },
        {
            "matcher": {"id": "byName", "options": "Anomaly"},
            "properties": [
                {"id": "color", "value": {"fixedColor": "#F2495C", "mode": "fixed"}},
                {"id": "custom.showPoints", "value": "always"},
                {"id": "custom.pointSize", "value": 8},
                {"id": "custom.lineWidth", "value": 0},
                {"id": "custom.drawStyle", "value": "points"},
                {"id": "custom.spanNulls", "value": False},
            ],
        },
        {
            "matcher": {"id": "byName", "options": "bb_upper"},
            "properties": bb_upper_props,
        },
        {
            "matcher": {"id": "byName", "options": "bb_lower"},
            "properties": [
                {"id": "color", "value": {"fixedColor": "dark-blue", "mode": "fixed"}},
                {"id": "custom.lineStyle", "value": {"dash": [10, 10], "fill": "dash"}},
                {"id": "displayName", "value": "Rolling Lower Bound (-3\u03c3)"},
            ],
        },
        {
            "matcher": {"id": "byName", "options": "rolling_mean"},
            "properties": [
                {"id": "color", "value": {"fixedColor": "dark-purple", "mode": "fixed"}},
                {"id": "custom.lineStyle", "value": {"dash": [10, 10], "fill": "dash"}},
            ],
        },
    ]


def build(
    panel_id: int,
    title: str,
    metric_name: str,
    station_name: str,
    datasource: dict,
    unit: str = "s",
    grid_pos: dict = None,
    line_width: int = 2,
    show_points: str = "never",
    value_color: str = "#37872D",
    bb_upper_show_points: bool = False,
    rolling_datasource_type: str = "grafana-postgresql-datasource",
) -> dict:
    """
    Build a complete SPC timeseries panel dict.

    Args:
        panel_id:                  Grafana panel id (must be unique within dashboard)
        title:                     Panel title shown in Grafana
        metric_name:               metric_name value used in SQL WHERE clause
        station_name:              station_name value used in SQL WHERE clause
        datasource:                dict with 'type' and 'uid' keys (used for Target A)
        unit:                      Grafana unit string (default 's' for seconds)
        grid_pos:                  dict with h/w/x/y (auto-assigned by layout.py if None)
        line_width:                default line width for the panel (default 2)
        show_points:               default show points mode (default 'never')
        value_color:               fixedColor for the 'value' series override (default '#37872D')
        bb_upper_show_points:      include custom.showPoints on bb_upper override (default False)
        rolling_datasource_type:   datasource type for Target B (default 'grafana-postgresql-datasource')
    """
    if grid_pos is None:
        grid_pos = {"h": 8, "w": 12, "x": 0, "y": 0}

    rolling_ds = {"type": rolling_datasource_type, "uid": datasource["uid"]}

    return {
        "id": panel_id,
        "type": "timeseries",
        "title": title,
        "pluginVersion": "11.4.0",
        "datasource": datasource,
        "gridPos": grid_pos,
        "fieldConfig": {
            "defaults": {
                "color": {"fixedColor": "#37872D", "mode": "fixed"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "",
                    "axisPlacement": "auto",
                    "barAlignment": 0,
                    "barWidthFactor": 0.6,
                    "drawStyle": "line",
                    "fillOpacity": 0,
                    "gradientMode": "none",
                    "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                    "insertNulls": False,
                    "lineInterpolation": "linear",
                    "lineWidth": line_width,
                    "pointSize": 5,
                    "scaleDistribution": {"type": "linear"},
                    "showPoints": show_points,
                    "spanNulls": False,
                    "stacking": {"group": "A", "mode": "none"},
                    "thresholdsStyle": {"mode": "off"},
                },
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "red", "value": 80},
                    ],
                },
                "unit": unit,
            },
            "overrides": _field_overrides(value_color, bb_upper_show_points),
        },
        "options": {
            "legend": {
                "calcs": [],
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"mode": "multi", "sort": "none"},
        },
        "targets": [
            {
                "datasource": datasource,
                "editorMode": "code",
                "format": "time_series",
                "rawQuery": True,
                "rawSql": _build_sql_main(station_name, metric_name),
                "refId": "A",
                "sql": _SQL_STUB,
            },
            {
                "datasource": rolling_ds,
                "editorMode": "code",
                "format": "table",
                "hide": False,
                "rawQuery": True,
                "rawSql": _build_sql_rolling(station_name, metric_name),
                "refId": "B",
                "sql": _SQL_STUB,
            },
        ],
    }
