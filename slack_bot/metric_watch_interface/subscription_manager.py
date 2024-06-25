from __future__ import annotations

from datetime import datetime

from slack_bot.metric_watch_interface.database import db, Metrics, Subscriptions


class SubscriptionManager:
    def list_metrics(self) -> list[str]:
        """List all metrics."""
        return [metric.name for metric in Metrics.objects_in(db)]

    def metric_exists(self, metric: str) -> bool:
        """Check if a metric exists."""
        return metric in self.list_metrics()

    def is_subscribed(self, user: str, metric: str) -> bool:
        """Check if a user is subscribed to a metric."""
        queryset = Subscriptions.objects_in(db).filter(user=user, metric=metric)
        return queryset.count() > 0

    def subscribe(self, user: str, metric: str) -> str:
        """Subscribe a user to a metric."""
        subscription = Subscriptions(
            user=user, metric=metric, subscribed_at=datetime.now(tz=db.server_timezone)
        )
        db.insert([subscription])

        return f"\nUser '{user}' subscribed to '{metric}' successfully."

    def unsubscribe(self, user: str, metric: str) -> str:
        """Unsubscribe a user from a metric."""
        Subscriptions.objects_in(db).filter(user=user, metric=metric).delete()

        return f"\nUser '{user}' unsubscribed from '{metric}' successfully."

    def sub_or_unsub(self, user: str, metric: str) -> str:
        """Subscribe or unsubscribe a user from a metric."""
        if self.is_subscribed(user, metric):
            return self.unsubscribe(user, metric)
        return self.subscribe(user, metric)
