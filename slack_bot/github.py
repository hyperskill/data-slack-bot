from __future__ import annotations

import requests
from typing import Any


class IssueCreationError(Exception):
    """Exception raised for errors in the issue creation process."""
    
    def __init__(self, status_code: int, error_message: str) -> None:
        self.status_code = status_code
        self.error_message = error_message
        super().__init__(
            f"Failed to create issue. Status code: {status_code}.\n"
            f"Error: {error_message}"
        )


class GitHub:
    def __init__(self, base_url: str | None, token: str | None) -> None:
        if base_url or token:
            self.base_url = base_url
            self.token = token
        else:
            raise ValueError("GitHub base URL and API token are required.")

    def create_issue(
        self, summary: str, description: str, repository: str
    ) -> dict[str, Any] | IssueCreationError:
        """Create an issue in GitHub repository.
        
        Args:
            summary (str): The title of the issue.
            description (str): The body of the issue.
            repository (str): The repository name to create the issue in.
            
        Returns:
            dict[str, Any] | IssueCreationError: The created issue data or an error.
        """
        # GitHub API URL format: https://api.github.com/repos/{owner}/{repo}/issues
        url = f"https://api.github.com/repos/hyperskill/{repository}/issues"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        payload = {
            "title": summary,
            "body": description
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        # GitHub returns 201 for created resources
        if response.status_code not in (200, 201):
            return IssueCreationError(response.status_code, response.text)

        return response.json()
