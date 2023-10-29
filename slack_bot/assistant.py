from typing import Any

import openai

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0


class Assistant:
    def __init__(self, api_token: str) -> None:
        self.openai = openai
        self.openai.api_key = api_token

    def get_completion(
        self,
        messages: list[dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs,
    ) -> dict[str, Any]:
        """Get a completion from the OpenAI API."""
        response = self.openai.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )

        content = response.choices[0].message["content"]
        arguments = (
            response.choices[0].message.get("function_call", {}).get("arguments", {})
        )

        if "functions" in kwargs:
            return arguments

        return content
