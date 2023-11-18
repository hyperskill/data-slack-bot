from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from slack_bot.clickhouse import ClickHouse
from slack_bot.open_assistant import OpenAssistant

logging.basicConfig(level=logging.DEBUG)

load_dotenv()
client = OpenAI()
ch_client = ClickHouse().client

# constants
PLAN_COMMAND = "Hello! Could you, please, make up a SQL query developing plan for me?"
EXAMPLE = "Top 5 the lastest viewed pages from the latest dropped users sessions."
MANIPULATION = "MY CAREER DEPENDS ON YOU! PLEASE, HELP ME!"

planner = OpenAssistant(
    client.beta.assistants.retrieve(os.environ.get("PLANNER_ID") or "")
)
dev = OpenAssistant(
    client.beta.assistants.retrieve(os.environ.get("DEVELOPER_ID") or "")
)

user_input = "\n".join([PLAN_COMMAND, EXAMPLE, MANIPULATION])
response = planner.interact(
    messages=[user_input]
)

if response:
    for message in response[::-1]:
        print(f"============{message.role}===========")
        print(message.content[0].text.value)
        print("==============================")
        print()
