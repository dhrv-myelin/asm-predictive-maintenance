def assign_grid_positions(panels, panel_width=12, panel_height=8, columns=24):
    x, y = 0, 0
    for panel in panels:
        panel["gridPos"] = {
            "h": panel_height,
            "w": panel_width,
            "x": x,
            "y": y,
        }
        x += panel_width
        if x >= columns:
            x = 0
            y += panel_height
    return panels
