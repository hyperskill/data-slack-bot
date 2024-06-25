from __future__ import annotations

from dotenv import load_dotenv
from infi.clickhouse_orm import DateTimeField, Model, StringField
from infi.clickhouse_orm.engines import MergeTree

load_dotenv(override=True)


class Metrics(Model):
    name = StringField()

    engine = MergeTree(partition_key=("name",), order_by=("name",))


class Subscriptions(Model):
    user = StringField()
    metric = StringField()
    subscribed_at = DateTimeField()

    engine = MergeTree(partition_key=("user",), order_by=("subscribed_at",))
