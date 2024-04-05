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
PATH_TO_DATASET = Path(
    PATH_TO_ROOT / "data" / "datasets" / "sql_queries.csv"
)
functions = DB(PATH_TO_ROOT / "data" / "functions")
send_sql_plan = functions["send_sql_plan.json"] or "{}"
prompts = DB(PATH_TO_ROOT / "data" / "prompts")
system_prompt: str = prompts["sql_query_plan"] or ""

if send_sql_plan == "" or system_prompt == "":
    raise ValueError("Please, check if you have functions and prompts in data folder.")


def convert_via_promt(
        targets: list[str], prompt: str, **kwargs
    ) -> list[str | None]:
    """Convert target into completion."""
    assistent = Assistant()
    result = []
    PATH_TO_ARTEFACTS = Path(__file__).parent / "artefacts"

    for i, query in enumerate(targets):
        if i:
            message = "Progress: " + str(i / len(targets))
            logging.info(message)

        response = None

        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(60)

            response = assistent.get_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                tools=kwargs.get("tools", None),
                tool_choice=kwargs.get("tool_choice", None)
            )
        except Exception:
            logging.exception("Exception occurred")
        finally:
            if response and isinstance(response, str):
                completion = json.loads(response)
                value = next(iter(completion.values()))

                if value:
                    logging.info("--- value: ---")
                    logging.info(value)
                    result.append(value)
                else:
                    logging.info("--- No completion ---")
                    logging.info(response)
                    result.append(None)
            else:
                raise ValueError("Response is not a string.")
        try:
            with PATH_TO_ARTEFACTS.open("w") as file:
                file.write(json.dumps(result))
        except Exception:
            logging.exception("Exception occurred")

    return result


def main() -> None:
    """Generate SQL query plans."""
    tools=[{
        "type": "function",
        "function": json.loads(send_sql_plan, strict=False)
    }]
    tool_choice={
        "type": "function",
        "function": {
            "name": "send_sql_plan"
        }
    }
    sql_df = pd.read_csv(PATH_TO_DATASET)
    queries = sql_df.sql.to_list()

    sql_df["plan"] = convert_via_promt(
        queries,
        system_prompt,
        tools=tools,
        tool_choice=tool_choice
    )
    sql_df.to_csv(PATH_TO_DATASET, index=False)


if __name__ == "__main__":
    main()
