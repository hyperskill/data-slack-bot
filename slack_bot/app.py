from __future__ import annotations

import openai
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from utils import (
    make_ai_response,
    OPENAI_API_KEY,
    SLACK_APP_TOKEN,
    SLACK_BOT_TOKEN,
)

app = App(token=SLACK_BOT_TOKEN)
openai.api_key = OPENAI_API_KEY


@app.event("app_mention")
@app.event("message")
def command_handler(body: dict[str, dict[str, str]], context: dict[str, str]) -> None:
    make_ai_response(app, body, context, openai)


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()  # type: ignore[no-untyped-call]
