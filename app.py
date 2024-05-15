from __future__ import annotations

import logging
import os

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

from slack_bot.helpers import run_with_the_best_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

app = App(token=SLACK_BOT_TOKEN)


@app.event("app_mention")
@app.event("message")
def command_handler(body: dict[str, dict[str, str]], context: dict[str, str]) -> None:
    try:
        logger.info("Received event: %s", body)
        run_with_the_best_model(app=app, body=body, context=context)
    except SlackApiError as e:
        logger.exception("Slack API error: %s", e.response["error"])
    except Exception:
        logger.exception("Found an error")


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()  # type: ignore[no-untyped-call]
