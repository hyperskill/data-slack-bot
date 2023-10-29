import json
import os
from pathlib import Path

from dotenv import load_dotenv

from slack_bot.assistant import Assistant
from slack_bot.clickhouse import ClickHouse
from slack_bot.db import DB

load_dotenv()

assistant = Assistant(os.environ.get("OPENAI_API_KEY"))
ch_client = ClickHouse().client


def update_history(sender: str, message: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Update the chat history with a new message"""
    history.append({"role": sender, "content": message})

    return history


prompts = DB(Path(__file__).parent.parent / "data" / "prompts")
shots = DB(Path(__file__).parent.parent / "data" / "shots")
functions = DB(Path(__file__).parent.parent / "data" / "functions")

SYSTEM = [
    {"role": "system", "content": prompts["developing"]},
]
SHOTS = [
    {"role": "user", "content": shots["dump_users"]},
    {"role": "assistant", "content": shots["dump_users.sql"]},
    {"role": "user", "content": shots["users_part"]},
    {"role": "assistant", "content": shots["users_part.sql"]},
]
PROBLEM = [
    {
        "role": "user",
        "content": "How much time users need to subscribe?",
    },
]

messages = SYSTEM + SHOTS + PROBLEM
phases = {"developing": {"result": assistant.get_completion(messages=messages)}}
messages = update_history("assistant", phases["developing"]["result"], messages)

# print(phases["developing"]["result"])

funcs = [functions["run_query"]]

for i in range(3):
    try:
        response = assistant.get_completion(
            messages=messages,
            functions=funcs,
            function_call={"name": "run_query"},
        )
        update_history("assistant", response, messages)

        result = str(ch_client.execute(json.loads(response, strict=False)["sql_query"]))
        phases["testing"]["result"] = result
        messages = update_history("assistant", result, messages)

    except Exception as e:
        result = "Error: " + str(e).split("Stack trace:")[0]
        messages = update_history("system", result, messages)
