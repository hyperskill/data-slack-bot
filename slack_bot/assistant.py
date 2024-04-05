from __future__ import annotations

from typing import Any, TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


DEFAULT_MODEL = "gpt-4-1106-preview"  # "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.1


class Assistant:
    def __init__(self) -> None:
        self.openai = client

    def get_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs,
    ) -> str | None:
        """Get a completion from the OpenAI API."""
        response = self.openai.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )
        content = response.choices[0].message.content
        arguments = response.choices[0].message.tool_calls[0].function.arguments

        if "tools" in kwargs:
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
