from __future__ import annotations

from typing import Any

import openai

DEFAULT_MODEL = "gpt-4-1106-preview"  # "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.1


class Assistant:
    def __init__(self, api_token: str | Any) -> None:
        self.openai = openai
        self.openai.api_key = api_token

    def get_completion(
        self,
        messages: list[dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs,
    ) -> str | dict[str, Any]:
        """Get a completion from the OpenAI API."""
        response = self.openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
            model=model, messages=messages, temperature=temperature, **kwargs
        )

        content = response.choices[0].message["content"]
        arguments = (
            response.choices[0].message.get("function_call", {}).get("arguments", {})
        )

        if "functions" in kwargs:
            return arguments

        return content


class Phase:
    def __init__(
        self,
        name: str,
        role: str | Any,
        shots: Any | list[dict[str, str]] = None,
        functions: Any = None,
    ) -> None:
        self.name = name
        self.role = role
        self.shots = shots
        self.functions = functions
        self.history: list[dict[str, str]] = []
        self.result = None

        self.update_history("system", role)

        if shots:
            self.history += shots

    def update_history(self, sender: str, message: str) -> None:
        """Update the chat history with a new message."""
        self.history.append({"role": sender, "content": message})
