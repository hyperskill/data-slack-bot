from __future__ import annotations

import json
from os import getenv
from pathlib import Path
from typing import Any

from slack_bot.db import DB
from slack_bot.youtrack import YouTrack
from dotenv import load_dotenv
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
load_dotenv()

YT_BASE_URL = getenv("YT_BASE_URL")
YT_API_TOKEN = getenv("YT_API_TOKEN")
prompts = DB(Path(__file__).parent / "prompts")
templates = DB(Path(__file__).parent / "templates")
functions = DB(Path(__file__).parent / "functions")


# define scenario for YouTrack issue submission
def submit_an_issue(message: str) -> Any | None:
    """Assist user in submitting an issue to YouTrack."""
    yt = YouTrack(
        base_url=YT_BASE_URL,
        token=YT_API_TOKEN,
    )

    sys_prompt = prompts["clarification"].replace("{template}", templates["yt_issue"])

    assistant = autogen.AssistantAgent(
        name="ClarifAI",
        system_message=sys_prompt,
        llm_config={
            "seed": 11,
            "config_list": config_list,
            "functions": [
                json.loads(functions["create_issue"]),
            ],
        },
        code_execution_config=False,
    )
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="ALWAYS",
    )

    user_proxy.register_function(
        function_map={
            "create_issue": yt.create_issue,
        }
    )

    autogen.ChatCompletion.start_logging()
    user_proxy.initiate_chat(assistant, message=message)

    # parse the logged history to extract the issue summary and description
    history = autogen.ChatCompletion.logged_history

    for key in history:
        chat = json.loads(key)
        for message in chat:
            if "function_call" in message:
                arguments = json.loads(message["function_call"]["arguments"])  # type: ignore[index]
                summary = arguments["summary"]
                description = arguments["description"]

    # create the issue
    return yt.create_issue(summary=summary, description=description)
