from __future__ import annotations

from os import environ as env

from clickhouse_driver import Client  # type: ignore[import]
from dotenv import load_dotenv

load_dotenv()


class ClickHouse:
    def __init__(self) -> None:
        self.host = env.get("CLICKHOUSE_HOST")
        self.user = env.get("CLICKHOUSE_USER")
        self.password = env.get("CLICKHOUSE_PASSWORD")
        self.client = Client(
            host=self.host,
            user=self.user,
            password=self.password,
        )
