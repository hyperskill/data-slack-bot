from __future__ import annotations

from typing import TYPE_CHECKING

import openai
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from slack_bot.utils import (
    N_CHUNKS_TO_CONCAT_BEFORE_UPDATING,
    num_tokens_from_messages,
    OPENAI_API_KEY,
    process_conversation_history,
    SLACK_APP_TOKEN,
    SLACK_BOT_TOKEN,
    update_chat,
    WAIT_MESSAGE,
)

if TYPE_CHECKING:
    from slack_sdk.web import SlackResponse

app = App(token=SLACK_BOT_TOKEN)
openai.api_key = OPENAI_API_KEY


def get_conversation_history(channel_id: str, thread_ts: str) -> SlackResponse:
    return app.client.conversations_replies(
        channel=channel_id, ts=thread_ts, inclusive=True
    )


@app.event("app_mention")
@app.event("message")
def command_handler(body: dict[str, dict[str, str]], context: dict[str, str]) -> None:
    try:
        channel_id = body["event"]["channel"]
        thread_ts = body["event"].get("thread_ts", body["event"]["ts"])
        bot_user_id = context["bot_user_id"]
        slack_resp = app.client.chat_postMessage(
            channel=channel_id, thread_ts=thread_ts, text=WAIT_MESSAGE
        )
        reply_message_ts = slack_resp["message"]["ts"]
        conversation_history = get_conversation_history(channel_id, thread_ts)
        messages = process_conversation_history(conversation_history, bot_user_id)
        num_tokens = num_tokens_from_messages(messages)
        print(f"Number of tokens: {num_tokens}")  # noqa: T201

        openai_response = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
            model="gpt-3.5-turbo", messages=messages, stream=True
        )

        response_text = ""
        ii = 0
        for chunk in openai_response:
            if chunk.choices[0].delta.get("content"):
                ii = ii + 1
                response_text += chunk.choices[0].delta.content
                if ii > N_CHUNKS_TO_CONCAT_BEFORE_UPDATING:
                    update_chat(app, channel_id, reply_message_ts, response_text)
                    ii = 0
            elif chunk.choices[0].finish_reason == "stop":
                update_chat(app, channel_id, reply_message_ts, response_text)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")  # noqa: T201
        app.client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f"I can't provide a response. Encountered an error:\n`\n{e}\n`",
        )


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()  # type: ignore[no-untyped-call]
