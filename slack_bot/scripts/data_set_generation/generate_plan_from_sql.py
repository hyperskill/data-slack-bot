# generate this .py script for SQL query plan generation:
# 1. read SQL queries from .csv dataset to pandas dataframe
# 2. for every query generate "SQL query plan" via slack_bot.assistant import Assistant
# 3. save dataframe to .csv file
from __future__ import annotations

from pathlib import Path

import pandas as pd

from slack_bot.assistant import Assistant
from slack_bot.db import DB

functions = DB(Path(__file__).parent / "data" / "functions")
send_sql_plan = functions["send_sql_plan"] or ""
prompts = DB(Path(__file__).parent / "data" / "prompts")
system_prompt: str = prompts["sql_query_plan"] or ""


def get_sql_query_plans(sql_query: str) -> str | None:
    """Get SQL query plans."""
    assistent = Assistant()
    result = assistent.get_completion(
        messages=[{
            "system": system_prompt,
            "user": sql_query
        }],
        functions=[send_sql_plan],
        function_call={"name": "send_sql_plan"}
    )

    if isinstance(result, dict) and "sql_query_plan" in result:
        return result["sql_query_plan"]

    return None


def main() -> None:
    """Generate SQL query plans."""
    sql_df = pd.read_csv("data/queries.csv")
    sql_df["plan"] = sql_df.query.apply(lambda x: get_sql_query_plans(x))
    sql_df.to_csv("data/queries.csv", index=False)


if __name__ == "__main__":
    main()
