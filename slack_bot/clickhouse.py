from clickhouse_driver import Client
from dotenv import dotenv_values


class ClickHouse:
    def __init__(self) -> None:
        self.host = dotenv_values(".env")["CLICKHOUSE_HOST"]
        self.user = dotenv_values(".env")["CLICKHOUSE_USER"]
        self.password = dotenv_values(".env")["CLICKHOUSE_PASSWORD"]
        self.client = Client(
            host=self.host,
            user=self.user,
            password=self.password,
        )
