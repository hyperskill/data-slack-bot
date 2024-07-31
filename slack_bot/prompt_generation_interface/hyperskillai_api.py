from __future__ import annotations

import json

import requests


class HyperskillAIAPI:
    __URL__ = "https://ai-provider.aks-hs-prod.int.hyperskill.org/chat-completion"

    def __init__(self, api_key: str, model: str, provider: str | None = None) -> None:
        self.__api_key = api_key
        self.__model = model
        self.__provider = provider

    def get_chat_completion(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        """Get a chat completion from the AI model.

        This method sends a request to the AI model API to
        generate a chat completion based on the given messages,
        temperature, and max tokens.

        Args:
            messages (list[dict[str, str]]): A list of messages
            temperature (float): The sampling temperature to
                use for generation. Higher values make the output
                more random, lower values make it more deterministic.
            max_tokens (int): The maximum number of tokens to generate
                in the chat completion.

        Returns:
            str: The generated chat completion content.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        system_prompt, user_prompts = self.get_system_prompt_from_messages(messages)
        system_message = system_prompt["content"]
        payload_dict = {
            "messages": user_prompts,
            "model": self.__model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if len(system_message) > 0:
            payload_dict["system"] = system_message

        if self.__provider is not None:
            payload_dict["provider"] = self.__provider
        payload = json.dumps(payload_dict)

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.__api_key,
        }

        response = requests.post(
            self.__URL__, headers=headers, data=payload, timeout=30
        )
        response.raise_for_status()
        return response.json()["content"]

    @staticmethod
    def get_system_prompt_from_messages(
        messages: list[dict[str, str]],
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """Extracts system prompts from messages.

        :param messages: list of messages with fields `role` and `content`
        :return: system_prompt object with fields `role` and `content`,
            user_prompts list of messages.
        """
        system_prompts = [
            message for message in messages if message["role"] == "system"
        ]
        system_prompt_content = "\n".join(
            message["content"] for message in system_prompts
        )
        system_prompt = {"role": "system", "content": system_prompt_content}

        user_prompts = [message for message in messages if message["role"] != "system"]
        return system_prompt, user_prompts
