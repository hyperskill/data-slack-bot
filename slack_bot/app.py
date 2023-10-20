from __future__ import annotations

import logging
import os
from typing import Any, TYPE_CHECKING

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

if TYPE_CHECKING:
    from slack_bolt.context.say import Say
    from slack_sdk.web import WebClient


load_dotenv(".vars")
logger = logging.getLogger(__name__)


# Initializes your app with your bot token and signing secret
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
app = App(token=SLACK_BOT_TOKEN)


# Add functionality here
@app.event("app_home_opened")
def update_home_tab(
    client: WebClient, event: dict[str, Any], logger: logging.Logger
) -> None:
    try:
        # views.publish is the method that your app uses to push a view to the Home tab
        client.views_publish(
            # the user that opened your app's app home
            user_id=event["user"],
            # the view object that appears in the app home
            view={
                "type": "home",
                "callback_id": "home_view",
                # body of the view
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Welcome to your _App's Home_* :tada:",
                        },
                    },
                    {"type": "divider"},
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "This button won't do much for now but you can set "
                            "up a listener for it using the `actions()` method "
                            "and passing its unique `action_id`. See an "
                            "example in the `examples` folder within your Bolt app.",
                        },
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Click me!"},
                            }
                        ],
                    },
                ],
            },
        )

    except Exception:
        logger.exception("Error publishing home tab")


@app.event("app_mention")
def event_test(body: str, say: Say, logger: logging.Logger) -> None:
    try:
        say("What's up?")
        logger.info(body)
    except Exception:
        logger.exception("Error responding to mention")


@app.event("message")
def event_test2(body: str, say: Say, logger: logging.Logger) -> None:
    try:
        say("What's up?")
        logger.info(body)
    except Exception:
        logger.exception("Error responding to mention")


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()  # type: ignore[no-untyped-call]
