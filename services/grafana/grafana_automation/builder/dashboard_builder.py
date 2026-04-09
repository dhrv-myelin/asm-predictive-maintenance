import importlib
import yaml

from .layout import assign_grid_positions
from features import annotations, rolling_window

TEMPLATE_REGISTRY = {
    "spc_timeseries": "templates.spc_timeseries",
}


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_template(panel_type: str):
    module_path = TEMPLATE_REGISTRY.get(panel_type)
    if not module_path:
        raise ValueError(f"Unknown panel type: '{panel_type}'. Available: {list(TEMPLATE_REGISTRY.keys())}")
    return importlib.import_module(module_path)


def _build_panels(config: dict) -> list:
    datasource = config["datasource"]
    station_name = config["station_name"]
    rw_cfg = config.get("features", {}).get("rolling_window", {})
    rolling_ds_type = rw_cfg.get("datasource_type", "grafana-postgresql-datasource")

    panels = []
    for i, panel_cfg in enumerate(config["panels"]):
        template = _load_template(panel_cfg["type"])
        panel_id = panel_cfg.get("id", i + 1)
        panel = template.build(
            panel_id=panel_id,
            title=panel_cfg["title"],
            metric_name=panel_cfg["metric_name"],
            station_name=station_name,
            datasource={"type": datasource["type"], "uid": datasource["uid"]},
            unit=panel_cfg.get("unit", "s"),
            line_width=panel_cfg.get("line_width", 2),
            show_points=panel_cfg.get("show_points", "never"),
            value_color=panel_cfg.get("value_color", "#37872D"),
            bb_upper_show_points=panel_cfg.get("bb_upper_show_points", False),
            rolling_datasource_type=rolling_ds_type,
        )
        panels.append(panel)

    return panels


def _build_templating(config: dict) -> dict:
    """
    Build the Grafana templating block.
    Always includes the datasource variable.
    If features.rolling_window.enabled is true, also adds the
    rolling_window custom variable with the options defined in config.
    """
    ds = config["datasource"]
    current = {}
    if ds.get("current_text") and ds.get("current_value"):
        current = {"text": ds["current_text"], "value": ds["current_value"]}

    variables = [
        {
            "current": current,
            "includeAll": False,
            "label": ds.get("label", "Data Source"),
            "name": "datasource",
            "options": [],
            "query": ds["type"],
            "refresh": 1,
            "type": "datasource",
        }
    ]

    rw_cfg = config.get("features", {}).get("rolling_window", {})
    if rw_cfg.get("enabled", False):
        options_list = rw_cfg.get("options", ["1 minute", "5 minutes", "10 minutes", "15 minutes", "30 minutes", "1 hour"])
        default_option = rw_cfg.get("default", "10 minutes")
        options = [
            {"selected": opt == default_option, "text": opt, "value": opt}
            for opt in options_list
        ]
        variables.append(
            {
                "current": {"text": default_option, "value": default_option},
                "name": "rolling_window",
                "options": options,
                "query": ",".join(options_list),
                "type": "custom",
            }
        )

    return {"list": variables}


def _resolve_time_range(config: dict) -> dict:
    time_cfg = config.get("time")
    if time_cfg:
        return time_cfg
    return {"from": "now-30d", "to": "now"}


def build_dashboard(config_path: str) -> dict:
    config = _load_config(config_path)
    dash_cfg = config["dashboard"]
    layout_cfg = config.get("layout", {})

    panels = _build_panels(config)
    panels = assign_grid_positions(
        panels,
        panel_width=layout_cfg.get("panel_width", 12),
        panel_height=layout_cfg.get("panel_height", 8),
        columns=layout_cfg.get("columns", 24),
    )

    annotation_list = annotations.build(config)
    time_range = _resolve_time_range(config)

    dashboard = {
        "annotations": {"list": annotation_list},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "links": [],
        "panels": panels,
        "preload": False,
        "refresh": dash_cfg.get("refresh", "1d"),
        "schemaVersion": dash_cfg.get("schemaVersion", 40),
        "tags": dash_cfg.get("tags", []),
        "templating": _build_templating(config),
        "time": time_range,
        "timepicker": {},
        "timezone": dash_cfg.get("timezone", "browser"),
        "title": dash_cfg["title"],
        "uid": dash_cfg["uid"],
        "version": dash_cfg.get("version", 1),
        "weekStart": "",
    }

    if "id" in dash_cfg:
        dashboard["id"] = dash_cfg["id"]

    return dashboard
