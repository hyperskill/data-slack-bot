from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

from slack_bot.clickhouse import ClickHouse
from slack_bot.open_assistant import OpenAssistant

if TYPE_CHECKING:
    from openai.types.beta.assistant_create_params import Tool


logging.basicConfig(level=logging.INFO)

load_dotenv()
client = OpenAI()
ch_client = ClickHouse().client

# constants
PLAN_COMMAND = "Hello! Could you, please, make up a SQL query developing plan for me?"
EXAMPLE = "Top 5 the lastest viewed pages from the latest dropped users sessions."
MANIPULATION = "MY CAREER DEPENDS ON YOU! PLEASE, HELP ME!"
RETRIEVAL: Tool = {"type": "retrieval"}
TOOLS: list[Tool] = [RETRIEVAL]

planner = OpenAssistant(
    client.beta.assistants.retrieve(os.environ.get("PLANNER_ID") or ""),
    tools=TOOLS
)
dev = OpenAssistant(
    client.beta.assistants.retrieve(os.environ.get("DEVELOPER_ID") or ""),
    tools=TOOLS
)

user_input = "\n".join([PLAN_COMMAND, EXAMPLE, MANIPULATION])
response = planner.interact(
    messages=[user_input]
)

if response:
    for message in response[::-1]:
        print(f"============{message.role}===========")
        print(message.content[0].text.value) # type: ignore[union-attr]
        print()

    planner_result = response[0].content[0].text.value # type: ignore[union-attr]

    response = dev.interact([planner_result])

    if response:
        for message in response[::-1]:
            print(f"============{message.role}===========")
            print(message.content[0].text.value)  # type: ignore[union-attr]
