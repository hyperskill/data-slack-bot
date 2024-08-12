from __future__ import annotations

import json
import logging
from os import getenv
from typing import Any

import requests
from dotenv import load_dotenv

from slack_bot.metric_watch_interface.constants import AIRFLOW_URL, DAG_ID

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class AirflowAPIError(Exception):
    """Custom exception for Airflow API errors."""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

class AirflowClient:
    def __init__(self, base_url: str, auth: tuple[str, str] | None = None) -> None:
        """Initialize the AirflowClient.

        :param base_url: Base URL of the Airflow instance, e.g., 'http://localhost:8080/api/v1'.
        :param auth: Tuple containing (username, password) for Basic Auth, if required.
        """
        self.base_url = base_url.rstrip("/")
        self.auth = auth

    def trigger_dag(
            self, dag_id: str, run_id: str | None = None,
            conf: dict[str, Any] | None = None
        ) -> dict[str, Any]:
        """Trigger a DAG run in Airflow.

        :param dag_id: The ID of the DAG to trigger.
        :param run_id: Optional, a specific ID for this DAG run.
        :param conf: Optional, a dictionary containing configuration for the DAG run.
        :return: Response from the Airflow API.
        :raises AirflowAPIError: If the API request fails.
        """
        url = f"{self.base_url}/dags/{dag_id}/dagRuns"
        payload: dict[str, Any] = {}

        if run_id:
            payload["dag_run_id"] = run_id

        if conf:
            payload["conf"] = conf

        headers = {
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                url=url,
                data=json.dumps(payload),
                headers=headers,
                auth=self.auth,
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise AirflowAPIError(f"Failed to trigger DAG '{dag_id}': {e}") from None


# Example usage:
if __name__ == "__main__":
    # Set your Airflow base URL and authentication details here.
    base_url = AIRFLOW_URL
    user = getenv("AIRFLOW_USER")
    password = getenv("AIRFLOW_PASSWORD")

    auth = (user, password) if user and password else None

    airflow_client = AirflowClient(base_url=base_url, auth=auth)

    # Trigger a DAG
    try:
        response = airflow_client.trigger_dag(
            dag_id=DAG_ID, conf={"user": "UNCMCAQ12"}
        )
        logging.info(response)
    except AirflowAPIError:
        logging.exception("Failed to trigger DAG.")
