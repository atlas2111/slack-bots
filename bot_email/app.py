import os
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from langchain import HuggingFaceHub, LLMChain, PromptTemplate

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_EMAIL_BOT_TOKEN = os.environ["SLACK_EMAIL_BOT_TOKEN"]
SLACK_EMAIL_SIGNING_SECRET = os.environ["SLACK_EMAIL_SIGNING_SECRET"]
SLACK_EMAIL_BOT_USER_ID = os.environ["SLACK_EMAIL_BOT_USER_ID"]

# Initialize your LLM model and template
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
repo_id = "tiiuae/falcon-7b-instruct"
falcon_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_new_tokens": 400})

# Initialize the Slack app
app = App(token=SLACK_EMAIL_BOT_TOKEN, signing_secret=SLACK_EMAIL_SIGNING_SECRET)

# Initialize the Flask app
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


# Define your EmailGenerator class
# Define your EmailGenerator class
class EmailGenerator:
    def generate_email(self, input_data, ):
        template = """You are a helpful assistant that drafts an {email} reply based on an a new email."""

        # Create a prompt template instance
        prompt = PromptTemplate(template=template, input_variables=["email"])

        # Create an LLMChain instance
        llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

        # Generate email content based on user input

        generated_email = llm_chain.run(input_data)

        return generated_email


# Create an instance of your EmailGenerator class
email_bot = EmailGenerator()


# Define your event handler and route as before
@app.event("app_mention")
def handle_mentions(body, say, client):
    text = body["event"]["text"]
    mention = f"<@{SLACK_EMAIL_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    say("Sure, I'll get right on that!")
    email_content = email_bot.generate_email(text)
    try:
        response = client.chat_postMessage(channel=body["event"]["channel"], text=email_content)
        print(response)
    except SlackApiError as e:
        print(f"Error sending response: {e.response['error']}")


@flask_app.route("/generate_email/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)


if __name__ == "__main__":
    flask_app.run(port=3000)
