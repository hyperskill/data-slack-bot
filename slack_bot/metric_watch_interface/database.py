from __future__ import annotations

from os import environ as env

from dotenv import load_dotenv
from infi.clickhouse_orm import Database, DateTimeField, Model, StringField
from infi.clickhouse_orm.engines import MergeTree

from slack_bot.metric_watch_interface.constants import METRIC_WATCH_DB

load_dotenv(override=True)


class Metrics(Model):
    name = StringField()

    engine = MergeTree(partition_key=("name",), order_by=("name",))


class Subscriptions(Model):
    user = StringField()
    metric = StringField()
    subscribed_at = DateTimeField()

    engine = MergeTree(partition_key=("user",), order_by=("subscribed_at",))


db = Database(
    db_name=METRIC_WATCH_DB,
    db_url=str(env.get("CLICKHOUSE_HOST_URL")),
    username=env.get("CLICKHOUSE_USER"),
    password=env.get("CLICKHOUSE_PASSWORD"),
)
# Create tables if not exist
db.create_table(Metrics)
db.create_table(Subscriptions)
