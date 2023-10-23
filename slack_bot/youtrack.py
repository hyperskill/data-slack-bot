from __future__ import annotations

from typing import Any

import requests


class IssueCreationError(Exception):
    """Exception raised for errors in the issue creation process."""

    def __init__(self, status_code: int, error_message: str) -> None:
        self.status_code = status_code
        self.error_message = error_message
        super().__init__(
            f"Failed to create issue. Status code: {status_code}.\n"
            f"Error: {error_message}"
        )


class YouTrack:
    def __init__(self, base_url: str | None, token: str | None) -> None:
        if base_url or token:
            self.base_url = base_url
            self.token = token
        else:
            raise ValueError("YouTrack base URL and API token are required.")

    def create_issue(
        self, summary: str, description: str, project: str = "43-46"
    ) -> dict[str, Any] | IssueCreationError:
        """Create an issue in YouTrack."""
        url = f"{self.base_url}/api/issues"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        payload = {
            "project": {"id": project},
            "summary": summary + " [created by AI Data Assistant]",
            "description": description,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:  # noqa: PLR2004
            return IssueCreationError(response.status_code, response.text)

        return response.json()
