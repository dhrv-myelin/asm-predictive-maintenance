

import psycopg2
import psycopg2.extras


def get_alert_volume_by_category(conn, start: str, end: str) -> list[dict]:
    """Alert count grouped by day and severity."""
    sql = f"""
        SELECT
            DATE_TRUNC('day', time) AS "Time",
            severity                AS "Severity",
            COUNT(*)                AS "Value"
        FROM alerts_unified
        WHERE time >= '{start}'
          AND time <  '{end}'
        GROUP BY 1, severity
        ORDER BY 1
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(row) for row in cur.fetchall()]


def get_anomaly_events(conn, start: str, end: str) -> list[dict]:
    """3σ anomaly events grouped by station and metric."""
    sql = f"""
        SELECT
            TO_CHAR(MAX(time) AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD HH24:MI:SS')::TEXT AS "Detected At",
            station                                                                         AS "Station",
            metric_name                                                                     AS "Metric",
            MAX(title)                                                                      AS "Remark",
            MAX(detail)                                                                     AS "Details",
            COUNT(*)                                                                        AS "Occurrences"
        FROM alerts_unified
        WHERE severity = 'anomaly'
          AND time >= '{start}'
          AND time <  '{end}'
        GROUP BY station, metric_name
        ORDER BY MAX(time) DESC
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(row) for row in cur.fetchall()]


def get_top_anomalous_metrics(conn, start: str, end: str) -> list[dict]:
    """Top 20 most frequently anomalous metrics."""
    sql = f"""
        SELECT
            metric_name  AS "Metric",
            COUNT(*)     AS "Anomaly Count"
        FROM alerts_unified
        WHERE severity = 'anomaly'
          AND time >= '{start}'
          AND time <  '{end}'
        GROUP BY metric_name
        ORDER BY "Anomaly Count" DESC
        LIMIT 20
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(row) for row in cur.fetchall()]


def get_anomaly_by_station(conn, start: str, end: str) -> list[dict]:
    """Top 20 stations by anomaly count."""
    sql = f"""
        SELECT
            station      AS "Station",
            COUNT(*)     AS "Anomaly Count"
        FROM alerts_unified
        WHERE severity = 'anomaly'
          AND time >= '{start}'
          AND time <  '{end}'
        GROUP BY station
        ORDER BY "Anomaly Count" DESC
        LIMIT 20
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(row) for row in cur.fetchall()]


def get_error_logs(conn, start: str, end: str) -> list[dict]:
    """Error-level log messages grouped by message."""
    sql = f"""
        SELECT
            TO_CHAR(MAX(timestamp) AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD HH24:MI:SS')::TEXT AS "Detected At",
            log_message                                                                          AS "Error Message",
            COUNT(*)                                                                             AS "Occurrences",
            'ERROR'                                                                              AS "Tags"
        FROM error_logs
        WHERE severity = 'error'
          AND timestamp >= '{start}'
          AND timestamp <  '{end}'
        GROUP BY log_message
        ORDER BY MAX(timestamp) DESC
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(row) for row in cur.fetchall()]


def get_warning_logs(conn, start: str, end: str) -> list[dict]:
    """Warning-level log messages grouped by message."""
    sql = f"""
        SELECT
            TO_CHAR(MAX(timestamp) AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD HH24:MI:SS')::TEXT AS "Detected At",
            log_message                                                                          AS "Error Message",
            COUNT(*)                                                                             AS "Occurrences",
            'WARNING'                                                                            AS "Tags"
        FROM error_logs
        WHERE severity = 'warning'
          AND timestamp >= '{start}'
          AND timestamp <  '{end}'
        GROUP BY log_message
        ORDER BY MAX(timestamp) DESC
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(row) for row in cur.fetchall()]


def get_maintenance_volume(conn, start: str, end: str) -> list[dict]:
    """Maintenance event count grouped by day."""
    sql = f"""
        SELECT
            DATE_TRUNC('day', time) AS "Time",
            COUNT(*)                AS "Maintenance Events"
        FROM alerts_unified
        WHERE severity = 'maintenance'
          AND time >= '{start}'
          AND time <  '{end}'
        GROUP BY 1
        ORDER BY 1
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(row) for row in cur.fetchall()]


# def get_maintenance_log(conn, start: str, end: str) -> list[dict]:
#     """Full maintenance event log ordered by most recent."""
#     sql = f"""
#         SELECT
#             TO_CHAR(time AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD HH24:MI:SS')::TEXT AS "Detected At",
#             metric_name                                                                AS "Metric",
#             title                                                                      AS "Status",
#             detail                                                                     AS "Detail"
#         FROM alerts_unified
#         WHERE severity = 'maintenance'
#           AND time >= '{start}'
#           AND time <  '{end}'
#         ORDER BY time DESC
#     """
#     with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
#         cur.execute(sql)
#         return [dict(row) for row in cur.fetchall()]