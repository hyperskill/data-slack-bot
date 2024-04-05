from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from generate_plan_from_sql import convert_via_promt

from slack_bot.db import DB

load_dotenv()
logging.basicConfig(level=logging.INFO)

PATH_TO_ROOT = Path(__file__).parent.parent.parent
PATH_TO_DATASET = Path(PATH_TO_ROOT / "data" / "datasets" / "sql_queries.csv")

functions = DB(PATH_TO_ROOT / "data" / "functions")
send_user_request = functions["send_user_request.json"] or "{}"
prompts = DB(PATH_TO_ROOT / "data" / "prompts")
system_prompt: str = prompts["user_request_from_sql_plan"] or ""

if send_user_request == "" or system_prompt == "":
    raise ValueError("Please, check if you have functions and prompts in data folder.")


def main() -> None:
    """Generate SQL query requests."""
    tools = [
        {"type": "function", "function": json.loads(send_user_request, strict=False)}
    ]
    tool_choice = {"type": "function", "function": {"name": "send_user_request"}}
    sql_df = pd.read_csv(PATH_TO_DATASET)
    plans = sql_df.plan.to_list()

    sql_df["request"] = convert_via_promt(
        plans, system_prompt, tools=tools, tool_choice=tool_choice
    )
    sql_df.to_csv(PATH_TO_DATASET, index=False)


if __name__ == "__main__":
    main()
