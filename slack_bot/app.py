from __future__ import annotations

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bot.helpers import (
    run_with_the_best_model,
    SLACK_APP_TOKEN,
    SLACK_BOT_TOKEN,
)

app = App(token=SLACK_BOT_TOKEN)


@app.event("app_mention")
@app.event("message")
def command_handler(body: dict[str, dict[str, str]], context: dict[str, str]) -> None:
    run_with_the_best_model(app=app, body=body, context=context)


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()  # type: ignore[no-untyped-call]
