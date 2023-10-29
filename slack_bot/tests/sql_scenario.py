import os
from pathlib import Path

from dotenv import load_dotenv

from slack_bot.db import DB
from slack_bot.assistant import Assistant

load_dotenv()

assistant = Assistant(os.environ.get("OPENAI_API_KEY"))


prompts = DB(Path(__file__).parent.parent / "data" / "prompts")
shots = DB(Path(__file__).parent.parent / "data" / "shots")
docs = DB(Path(__file__).parent.parent / "data" / "docs")

SYSTEM = [
    {"role": "system", "content": prompts["developing"]},
]
SHOTS = [
    {"role": "user", "content": shots["dump_users"]},
    {"role": "assistant", "content": shots["dump_users.sql"]},
]
PROBLEM = [
    {
        "role": "user",
        "content": "How many users completed submission on their registration date on 2023-01-01?",
    },
]

messages = SYSTEM + SHOTS + PROBLEM
phases = {"developing": {"result": assistant.get_completion(messages=messages)}}

print(phases["developing"]["result"])
