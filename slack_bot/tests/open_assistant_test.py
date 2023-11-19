from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from slack_bot.db import DB
from slack_bot.open_assistant import OpenAssistant

logging.basicConfig(level=logging.INFO)

load_dotenv()
client = OpenAI()
shots = DB(Path(__file__).parent.parent / "data" / "shots")

dev_shots: list[str] = [
        shots["dump_users"] or "",
        shots["dump_users.sql"] or "",
        shots["users_part"] or "",
        shots["users_part.sql"] or "",
    ]

# constants
EXAMPLE = """Hello! Could you, please, help me?\n
I need top 5 the lastest viewed pages from the latest dropped users sessions.\n
MANIPULATION = "MY CAREER DEPENDS ON YOU! PLEASE, HELP ME!"""

dev = OpenAssistant(
    client.beta.assistants.retrieve(os.environ.get("DEVELOPER_ID") or "")
)

user_input = input("Enter your message: ")
if user_input == "":
    user_input = EXAMPLE

messages = [*dev_shots, user_input]

response = dev.interact(
    messages=messages
)

if response:
    for message in response[::-1]:
        print(f"============{message.role}===========")
        print(message.content[0].text.value) # type: ignore[union-attr]
        print()
