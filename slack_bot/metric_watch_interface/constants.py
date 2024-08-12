from __future__ import annotations

DB_URL = "http://{host}:8123"
METRIC_WATCH_DB = "metric_watch"
MENU = """
1. List all metrics
2. Unsubscribe / Subscribe to a metric
3. Show CLI help

Or type run command + arguments from the help menu (see option 3)

Enter your choice:
"""
METRIC_DOES_NOT_EXIST = "\nError: Metric '{metric}' does not exist."
AIRFLOW_URL = "https://airflow.int.hyperskill.org/api/v1"
DAG_ID = "metric_watch_custom"  # metric_watch_search
METRIC_WATCH__DOC = """
```
Anomaly Detection CLI

Usage:
  main.py [--start=<start>] [--end=<end>] [--metric=<metric>]
          [--granularity=<granularity>] [--window=<window>] [--sigma=<sigma>]
          [--slice=<slice>]

Options:
  --start=<start>                   Start date for the anomaly search
  --end=<end>                       End date for the anomaly search
  --metric=<metric>                 Metric name from metrics_hierarchy.yml
  --granularity=<granularity>       Granularity name (hour, day, week, month)
  --window=<window>                 Window size – days ago from a specific date
  --sigma=<sigma>                   Sigma value – number of std errors to trigger an alert
  --slice=<slice>                   Slice value – slice values from metrics_hierarchy.yml

Examples:
    python main.py --start=2024-01-01 --end=2024-04-01  `: check all metrics for the date range`
    python main.py --start=2024-01-01 --end=2024-04-01 --metric=new_registered_users
    python main.py --metric=new_registered_users --granularity=hour --window=168 --sigma=3 --slice=web
    python main.py --metric=new_registered_users --granularity=hour --window=168 --sigma=3
    python main.py --metric=new_registered_users --granularity=hour
    python main.py --metric=new_registered_users  `: check all metric granularities`
    python main.py --slice=Android  `: filter all metrics from the configuration that have the slice Android`
    python main.py  `: run the anomaly search for all metrics from the configuration`
```
"""
