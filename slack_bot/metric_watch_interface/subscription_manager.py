from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from slack_bot.metric_watch_interface.database import Metrics, Subscriptions

if TYPE_CHECKING:
    from infi.clickhouse_orm import Database


class SubscriptionManager:
    def __init__(self, db: Database) -> None:
        self.db = db

    def list_metrics(self) -> list[str]:
        """List all metrics."""
        return [metric.name for metric in Metrics.objects_in(self.db)]

    def metric_exists(self, metric: str) -> bool:
        """Check if a metric exists."""
        return metric in self.list_metrics()

    def is_subscribed(self, user: str, metric: str) -> bool:
        """Check if a user is subscribed to a metric."""
        queryset = Subscriptions.objects_in(self.db).filter(user=user, metric=metric)
        return queryset.count() > 0

    def subscribe(self, user: str, metric: str) -> str:
        """Subscribe a user to a metric."""
        subscription = Subscriptions(
            user=user,
            metric=metric,
            subscribed_at=datetime.now(tz=self.db.server_timezone),
        )
        self.db.insert([subscription])

        return f"\n:white_check_mark: Subscribed to `{metric}`."

    def unsubscribe(self, user: str, metric: str) -> str:
        """Unsubscribe a user from a metric."""
        Subscriptions.objects_in(self.db).filter(user=user, metric=metric).delete()

        return f"\n:warning: Warning: Unsubscribed from `{metric}`!"

    def sub_or_unsub(self, user: str, metric: str) -> str:
        """Subscribe or unsubscribe a user from a metric."""
        if self.is_subscribed(user, metric):
            return self.unsubscribe(user, metric)
        return self.subscribe(user, metric)
