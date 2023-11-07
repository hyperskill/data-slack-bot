from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING

import openai
import tiktoken
from dotenv import load_dotenv
from trafilatura import extract, fetch_url
from trafilatura.settings import use_config

from assistant import Assistant, Phase
from clickhouse import ClickHouse
from db import DB
from youtrack import YouTrack

load_dotenv()


if TYPE_CHECKING:
    from slack_bolt import App

newconfig = use_config()
newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
YT_API_TOKEN = os.environ.get("YT_API_TOKEN")

data = DB(Path(__file__).parent / "data")
prompts = DB(Path(__file__).parent / "data" / "prompts")
templates = DB(Path(__file__).parent / "data" / "templates")
functions = DB(Path(__file__).parent / "data" / "functions")
shots = DB(Path(__file__).parent / "data" / "shots")

WAIT_MESSAGE = "Got your request. Please wait."
MAX_TOKENS = 8192
BEST_MODEL = "gpt-4-1106-preview"
MODEL = "gpt-4"
projects = json.loads(data["yt_projects.json"])  # type: ignore[arg-type]
projects_shortnames = [project["shortName"].lower() for project in projects]
AN_COMMAND = "an"
YT_COMMAND = "yt"
SQL_COMMAND = "sql"
YT_BASE_URL = "https://vyahhi.myjetbrains.com/youtrack"

openai.api_key = OPENAI_API_KEY


def extract_url_list(text: str) -> list[str] | None:
    url_pattern = re.compile(
        r"<(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)>"
    )
    url_list = url_pattern.findall(text)

    return url_list if len(url_list) > 0 else None


def augment_user_message(user_message: str, url_list: list[str]) -> str:
    all_url_content = ""

    for url in url_list:
        downloaded = fetch_url(url)
        url_content = extract(downloaded, config=newconfig)
        user_message = user_message.replace(f"<{url}>", "")
        all_url_content = (
            all_url_content + f' Contents of {url} : \n """ {url_content} """'
        )

    return user_message + "\n" + all_url_content


# From https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(
    messages: list[dict[str, str]], model: str = "gpt-4"
) -> Any:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")  # noqa: T201
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo":
        # print(
        #     "Warning: gpt-3.5-turbo may change over time. "  # noqa: ERA001
        #     "Returning num tokens assuming gpt-3.5-turbo-0301."  # noqa: ERA001

        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")

    elif model == "gpt-4":  # noqa: RET505
        # print(
        #     "Warning: gpt-4 may change over time. "  # noqa: ERA001
        #     "Returning num tokens assuming gpt-4-0314."  # noqa: ERA001

        return num_tokens_from_messages(messages, model="gpt-4-0314")

    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted

    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md
            for information on how messages are converted to tokens."""
        )

    num_tokens = 0

    for message in messages:
        num_tokens += tokens_per_message

        for key, value in message.items():
            num_tokens += len(encoding.encode(value))

            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    return num_tokens


def clean_message_text(message_text: str, role: str, bot_user_id: str) -> str:
    if (f"<@{bot_user_id}>" in message_text) or (role == "assistant"):
        message_text = message_text.replace(f"<@{bot_user_id}>", "").strip()

    return message_text.strip()


def process_message(message: dict[str, str], bot_user_id: str, role: str) -> str:
    message_text = message["text"]

    if role == "user":
        url_list = extract_url_list(message_text)

        if url_list:
            message_text = augment_user_message(message_text, url_list)

    return clean_message_text(message_text, role, bot_user_id)


def process_conversation(
    conversation_messages: list[dict[str, str]], bot_user_id: str
) -> list[dict[str, str]]:
    conversation_messages.pop()  # remove WAIT_MESSAGE
    messages: list[dict[str, str]] = []
    """Transforms Slack messages into an openai API messages.
    Also, it inserts system message for clarification or YouTrack
    issue submission scenarios."""

    for message in conversation_messages:
        cleaned_message = message["text"].replace(f"<@{bot_user_id}>", "").strip()

        if cleaned_message in projects_shortnames:
            template = templates[cleaned_message] or templates[YT_COMMAND]

            system = prompts["clarification"].replace(  # type: ignore[union-attr]
                "{{template}}", template  # type: ignore[arg-type]
            )
            messages.append({"role": "system", "content": system})
            continue

        role = "assistant" if message["user"] == bot_user_id else "user"
        message_text = process_message(message, bot_user_id, role)

        if message_text:
            messages.append({"role": role, "content": message_text})

    return messages


def submit_issue(messages: list[dict[str, str]], project_id: str, model: str) -> str:
    funcs = [json.loads(functions["create_issue"])]  # type: ignore[arg-type]
    openai_response = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
        model=model,
        messages=messages,
        functions=funcs,
        function_call={"name": "create_issue"},
    )

    arguments = json.loads(
        openai_response.choices[0]
        .message.get("function_call", {})
        .get("arguments", {}),
        strict=False,
    )

    yt = YouTrack(
        base_url=YT_BASE_URL,
        token=YT_API_TOKEN,
    )

    response_text = yt.create_issue(
        summary=arguments["summary"],
        description=arguments["description"],
        project=project_id,
    )

    if isinstance(response_text, Exception):
        raise response_text

    return f"{YT_BASE_URL}/issue/{response_text['id']}"


def generate_sql(problem: str, model: str) -> str:
    assistant = Assistant(os.environ.get("OPENAI_API_KEY"))
    ch_client = ClickHouse().client

    dev_shots = [
        {"role": "user", "content": shots["dump_users"]},
        {"role": "assistant", "content": shots["dump_users.sql"]},
        {"role": "user", "content": shots["users_part"]},
        {"role": "assistant", "content": shots["users_part.sql"]},
    ]
    funcs = [json.loads(functions["run_query.json"])]  # type: ignore[arg-type]
    phases = {
        "developing": Phase(
            name="developing",
            role=prompts["developing"],
            shots=dev_shots,
            functions=funcs,
        ),
        "testing": Phase(
            name="testing",
            role=prompts["testing"],
            functions=funcs,
        ),
    }

    # developing phase
    phase = phases["developing"]
    phase.update_history("user", problem)
    completion = assistant.get_completion(
        model=model,
        messages=phases["developing"].history,
        functions=phase.functions,
        function_call={"name": "run_query"},
    )
    phase.result = json.loads(completion, strict=False)["sql_query"]  # type: ignore[arg-type]  # noqa: E501

    try:
        ch_client.execute(phase.result)
    except Exception as e:  # noqa: BLE001
        result = "Error: " + str(e).split("Stack trace:")[0]
        print(result)  # noqa: T201
        return (
            f"I am sorry, but I can't execute resulted query. Encountered an error:\n"
            f"```{result}```\n"
            f"```{phase.result}```"
        )
    else:
        return (
            f"Please, pay attention, that I am only a bot and I can't guarantee "
            f"that the resulted query is correct. Please, check it manually.\n"
            f"```{phase.result}```"
        )


def make_ai_response(
    app: App,
    body: dict[str, dict[str, str]],
    context: dict[str, str],
    model: str = BEST_MODEL,
) -> None:
    try:
        channel_id = body["event"]["channel"]
        thread_ts = body["event"].get("thread_ts", body["event"]["ts"])
        bot_user_id = context["bot_user_id"]

        slack_resp = app.client.chat_postMessage(
            channel=channel_id, thread_ts=thread_ts, text=WAIT_MESSAGE
        )
        reply_message_ts = slack_resp["message"]["ts"]

        conversation = app.client.conversations_replies(
            channel=channel_id, ts=thread_ts, inclusive=True
        )["messages"]
        messages = process_conversation(conversation, bot_user_id)

        # remove warning messages
        messages = [
            message
            for message in messages
            if not (message["content"].split(" ")[0] == "Sorry,")
            & (message["content"].split(" ")[-1] == "ignored.")
        ]

        num_tokens = num_tokens_from_messages(messages)
        # print(f"Number of tokens: {num_tokens}")  # noqa: ERA001

        if num_tokens > MAX_TOKENS * 0.95:
            app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=f"Sorry, you are using more than 95 % of tokens limit: "
                f"{num_tokens} / {MAX_TOKENS * 0.95}.\n"
                f"Some of the oldest messages will be ignored.",
            )

        while num_tokens_from_messages(messages) > MAX_TOKENS * 0.95:
            messages.pop(0)

        last_msg = messages[-1]
        last_msg_content = last_msg["content"]
        maybe_command = last_msg_content.split(" ")[0]

        if (last_msg["role"] == "user") & (maybe_command == YT_COMMAND):
            maybe_short_name = last_msg_content.split(" ")[-1]

            if maybe_short_name in projects_shortnames:
                project_id = next(
                    project
                    for project in projects
                    if project["shortName"].lower() == maybe_short_name
                )["id"]
            else:
                project_id = "43-46"

            template = templates[maybe_short_name] or templates[YT_COMMAND]
            messages.pop()  # remove YT_COMMAND
            messages.append(
                {
                    "role": "system",
                    "content": prompts[
                        "clarification"
                    ].replace(  # type: ignore[union-attr]
                        "{{template}}", template  # type: ignore[arg-type]
                    ),
                }
            )
            response_text = submit_issue(
                messages=messages, project_id=project_id, model=model
            )
        elif (last_msg["role"] == "user") & (maybe_command == SQL_COMMAND):
            response_text = generate_sql(
                problem=last_msg_content[len(SQL_COMMAND) + 1 :],
                model=model,
            )
        else:
            completion = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
                model=model, messages=messages
            )
            response_text = completion.choices[0].message["content"]

        app.client.chat_update(
            channel=channel_id, ts=reply_message_ts, text=response_text
        )
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")  # noqa: T201
        app.client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f"I can't provide a response. Encountered an error:\n`\n{e}\n`",
        )
        traceback.print_exc()


def run_with_the_best_model(**kwargs) -> None:
    try:
        make_ai_response(**kwargs)
    except Exception:  # noqa: BLE001
        make_ai_response(model=MODEL, **kwargs)
