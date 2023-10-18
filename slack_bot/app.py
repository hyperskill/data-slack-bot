from dotenv import load_dotenv
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

# Initializes your app with your bot token and signing secret
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET
)


# Add functionality here
@app.event("app_home_opened")
def update_home_tab(client, event, logger):
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
                    "text": "*Welcome to your _App's Home_* :tada:"
                  }
                },
                {
                  "type": "divider"
                },
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "This button won't do much for now but you can set "
                            "up a listener for it using the `actions()` method "
                            "and passing its unique `action_id`. See an "
                            "example in the `examples` folder within your Bolt app."
                  }
                },
                {
                  "type": "actions",
                  "elements": [
                    {
                      "type": "button",
                      "text": {
                        "type": "plain_text",
                        "text": "Click me!"
                      }
                    }
                  ]
                }
              ]
            }
          )

    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.event("app_mention")
def event_test(body, say, logger):
    try:
        say("What's up?")
        logger.info(body)
    except Exception as e:
        logger.error(f"Error responding to mention: {e}")


@app.event("message")
def event_test(body, say, logger):
    try:
        say("What's up?")
        logger.info(body)
    except Exception as e:
        logger.error(f"Error responding to mention: {e}")


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
