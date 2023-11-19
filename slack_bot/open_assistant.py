from __future__ import annotations

import logging
from time import sleep
from typing import Literal, TYPE_CHECKING

from openai import OpenAI

from slack_bot.exceptions import NoRunError, NoThreadError

if TYPE_CHECKING:
    from openai.types.beta.assistant import Assistant
    from openai.types.beta.assistant_create_params import Tool
    from openai.types.beta.thread import Thread
    from openai.types.beta.threads import Run, ThreadMessage

client = OpenAI()

SECONDS = 10
INFO_MESSAGE = f"Run not completed yet. Waiting {SECONDS} seconds..."

class OpenAssistant:
    def __init__(
        self,
        assistant: Assistant,
        model: str = "gpt-3.5-turbo",
        required_actions: list[str] | None=None,
        tools: list[Tool] | None=None,
    ) -> None:
        self.model = model,
        self.required_actions = required_actions
        self.tools = tools
        self.assistant = assistant
        self.thread: Thread | None=None
        self.message: ThreadMessage | None= None
        self.run: Run | None=None

    def create_thread(self) -> None:
        """Creates a new thread using OpenAI beta threads API.

        It does not return any value.
        """
        self.thread = client.beta.threads.create()

    def add_message_to_thread(
            self,
            content: str,
            role: Literal["user"],
            thread_id: str
        ) -> None:
        """Adds a new message to the thread."""
        try:
            self.message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content
            )
        except Exception:
            logging.exception("An error occurred while adding a message to the thread.")

    def create_run(self, **kwargs) -> None:
        """Creates a new run using OpenAI beta threads API."""
        if "tools" not in kwargs:
            kwargs["tools"] = self.tools
        if not self.thread:
            raise NoThreadError

        self.run = client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            **kwargs
        )

    def retrieve_run(self) -> Run:
        """Gets the run status."""
        if not self.thread:
            raise NoThreadError
        if not self.run:
            raise NoRunError

        return client.beta.threads.runs.retrieve(
            thread_id=self.thread.id,
            run_id=self.run.id
        )

    def get_thread_messages(self) -> list[ThreadMessage]:
        """Gets the thread messages."""
        if not self.thread:
            raise NoThreadError

        messages = client.beta.threads.messages.list(thread_id=self.thread.id)
        return messages.data

    def interact(
            self,
            messages: list[str] | None=None,
            **kwargs
        ) -> list[ThreadMessage] | None:
        """Interacts with the assistant."""
        if not self.thread:
            self.create_thread()
            if not self.thread:
                raise NoThreadError

        if messages:
            for message in messages:
                self.add_message_to_thread(
                    content=message,
                    role="user",
                    thread_id=self.thread.id
                )

        self.create_run(**kwargs)

        for _ in range(10):
            sleep(SECONDS)
            self.run = self.retrieve_run()

            if self.run.status == "completed":
                return self.get_thread_messages()

            logging.info(INFO_MESSAGE)

        return None
