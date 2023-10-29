import json
import os
from pathlib import Path

from dotenv import load_dotenv

from slack_bot.assistant import Assistant, Phase
from slack_bot.clickhouse import ClickHouse
from slack_bot.db import DB

load_dotenv()

assistant = Assistant(os.environ.get("OPENAI_API_KEY"))
ch_client = ClickHouse().client

prompts = DB(Path(__file__).parent.parent / "data" / "prompts")
shots = DB(Path(__file__).parent.parent / "data" / "shots")
functions = DB(Path(__file__).parent.parent / "data" / "functions")

dev_shots = [
    {"role": "user", "content": shots["dump_users"]},
    {"role": "assistant", "content": shots["dump_users.sql"]},
    {"role": "user", "content": shots["users_part"]},
    {"role": "assistant", "content": shots["users_part.sql"]},
]
problem = [
    {
        "role": "user",
        "content": "How much time users need to subscribe?",
    },
]
testing_funcs = [functions["run_query"]]
phases = {
    "developing": Phase(
        name="developing",
        role=prompts["developing"],
        shots=dev_shots,
    ),
    "testing": Phase(
        name="testing",
        role=prompts["testing"],
        functions=testing_funcs,
    ),
}


phase = phases["developing"]
phase.result = assistant.get_completion(
    messages=phases["developing"].history,
)

print(phase.result)

phase = phases["testing"]
phase.update_history("user", phases["developing"].result)

for _ in range(3):
    try:
        response = assistant.get_completion(
            messages=phase.history,
            functions=phase.functions,
            function_call={"name": "run_query"},
        )

        phase.result = str(
            ch_client.execute(json.loads(response, strict=False)["sql_query"])
        )

    except Exception as e:
        result = "Error: " + str(e).split("Stack trace:")[0]
        print(result)
        phase.update_history("system", result)
