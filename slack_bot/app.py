from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from data_assistant.scenarios import submit_an_issue

if TYPE_CHECKING:
    from slack_bolt.context.say import Say


load_dotenv()
logger = logging.getLogger(__name__)


# Initializes your app with your bot token and signing secret
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
app = App(token=SLACK_BOT_TOKEN)


@app.event("message")
def event_test2(body: str, say: Say, logger: logging.Logger) -> None:
    try:
        message = body["event"]["text"]
        reply = str(submit_an_issue(message))
        say("I am thinking...")
        say(reply)
        logger.info(body)
    except Exception:
        logger.exception("Error responding to mention")


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()  # type: ignore[no-untyped-call]
