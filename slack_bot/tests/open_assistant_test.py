from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

from slack_bot.db import DB
from slack_bot.open_assistant import OpenAssistant

if TYPE_CHECKING:
    from openai.types.beta.threads import ThreadMessage

logging.basicConfig(level=logging.INFO)

load_dotenv()
client = OpenAI()


def print_messages(response: list[ThreadMessage]) -> None:
    """Prints the messages in the response list in reverse order.

    Args:
        response (list[ThreadMessage]): The list of messages to be printed.
    """
    for message in response[::-1]:
        print(f"============{message.role}===========")  # noqa: T201
        print(message.content[0].text.value)  # type: ignore[union-attr] # noqa: T201
        print()  # noqa: T201


shots = DB(Path(__file__).parent.parent / "data" / "shots")

dev_shots: list[str] = [
    f"EXAMPLE 1:\n{shots['dump_users.sql']}",
    f"EXAMPLE 2:\n{shots['users_part.sql']}",
]

# constants
EXAMPLE = """Hello! Could you, please, help me?\n
I need top 5 the lastest viewed pages from the latest dropped users sessions.\n
MY CAREER DEPENDS ON YOU! PLEASE, HELP ME!"""

plan = OpenAssistant(
    client.beta.assistants.retrieve(os.environ.get("PLANNER_ID") or "")
)
dev = OpenAssistant(
    client.beta.assistants.retrieve(os.environ.get("DEVELOPER_ID") or "")
)

user_input = input("Enter your message: ")
if user_input == "":
    user_input = EXAMPLE

response = plan.interact(
    messages=[user_input],
    model="gpt-3.5-turbo-1106",
)

if response:
    print_messages(response)
    result = response[0].content[0].text.value  # type: ignore[union-attr]

    messages = [*dev_shots, result]

    response = dev.interact(messages=messages)

    if response:
        print_messages(response)
