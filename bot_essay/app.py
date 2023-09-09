import os
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_ESSAY_BOT_TOKEN = os.environ["SLACK_ESSAY_BOT_TOKEN"]
SLACK_ESSAY_SIGNING_SECRET = os.environ["SLACK_ESSAY_SIGNING_SECRET"]
SLACK_ESSAY_BOT_USER_ID = os.environ["SLACK_ESSAY_BOT_USER_ID"]


def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        slack_client = WebClient(token=SLACK_ESSAY_BOT_TOKEN)
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error getting bot user ID: {e}")
        return None


# Initialize your LLM model and template
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
repo_id = "tiiuae/falcon-7b-instruct"
falcon_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_new_tokens": 5000})

# Initialize the Slack app
app = App(token=SLACK_ESSAY_BOT_TOKEN, signing_secret=SLACK_ESSAY_SIGNING_SECRET)

# Initialize the Flask app
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


class EssayGenerator:
    def generate_essay(self, input_text):
        template = "I want you to act as an essay writer. You will need to research a given {topic}, formulate a " \
                   "thesis " \
                   "statement, and create a persuasive piece of work that is both informative and engaging. "

        # Create a prompt template instance
        prompt = PromptTemplate(template=template, input_variables=["topic"])

        # Create an LLMChain instance
        llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

        # Generate essay content based on user input
        generated_essay = llm_chain.run(input_text)

        return generated_essay


essay_bot = EssayGenerator()  # Initialize your EssayGenerator class


@app.event("app_mention")
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    text = body["event"]["text"]

    mention = f"<@{SLACK_ESSAY_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    say("Sure, I'll get right on that!")
    generated_essay = essay_bot.generate_essay(text)
    say(generated_essay)  # Respond with the generated essay


@flask_app.route("/generate_essay/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)


# Run the Flask app
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=8000)
