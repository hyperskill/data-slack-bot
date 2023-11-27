# generate this .py script for SQL query plan generation:
# 1. read SQL queries from .csv dataset to pandas dataframe
# 2. for every query generate "SQL query plan" via slack_bot.assistant import Assistant
# 3. save dataframe to .csv file
from __future__ import annotations

import json
import logging
import signal
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from slack_bot.assistant import Assistant
from slack_bot.db import DB

load_dotenv()


def handler(signum, frame):
    raise Exception("Time is up!")

logging.basicConfig(level=logging.INFO)

PATH_TO_ROOT = Path(__file__).parent.parent.parent
functions = DB(PATH_TO_ROOT / "data" / "functions")
send_sql_plan = functions["send_sql_plan.json"] or "{}"
prompts = DB(PATH_TO_ROOT / "data" / "prompts")
system_prompt: str = prompts["sql_query_plan"] or ""
PATH_TO_DATASET = Path(
    PATH_TO_ROOT / "data" / "datasets" / "sql_queries.csv"
)

if send_sql_plan == "" or system_prompt == "":
    raise ValueError("Please, check if you have functions and prompts in data folder.")


def get_sql_query_plans(sql_queries: list[str]) -> list[str | None]:
    """Get SQL query plans."""
    assistent = Assistant()
    result = []

    for i, query in enumerate(sql_queries):
        if i:
            message = "Progress: " + str(i / len(sql_queries))
            logging.info(message)

        response = None

        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(60)

            response = assistent.get_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                tools=[{
                    "type": "function",
                    "function": json.loads(send_sql_plan, strict=False)
                }],
                tool_choice={
                    "type": "function",
                    "function": {
                        "name": "send_sql_plan"
                    }
                }
            )
        except Exception:
            logging.exception("Exception occurred")
        finally:
            if response and isinstance(response, str):
                plan = json.loads(response)
                if "sql_query_plan" in plan:
                    plan = plan["sql_query_plan"]
                    logging.info(plan)
                    result.append(plan)
            else:
                result.append(None)

    return result


def main() -> None:
    """Generate SQL query plans."""
    sql_df = pd.read_csv(PATH_TO_DATASET)
    queries = sql_df.sql.to_list()
    sql_df["plan"] = get_sql_query_plans(queries)
    sql_df.to_csv(PATH_TO_DATASET, index=False)


if __name__ == "__main__":
    main()
