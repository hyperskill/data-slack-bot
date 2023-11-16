import openai

from openai import OpenAI

client = OpenAI()

class OpenAssistant:
    def __init__(self, assistant_name, model, tools=None):
        self.assistant_name = assistant_name
        self.model = model
        self.tools = tools if tools else []
        self.assistant_id = self.create_assistant()

    def create_assistant(self):
        response = client.beta.assistants.create(
            name=self.assistant_name,
            model=self.model,
            tools=self.tools
        )
        return response.id

    def create_thread(self):
        response = client.beta.threads.create()
        return response.id

    def add_message_to_thread(self, thread_id, message):
        response = openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )
        return response

    def run_assistant(self, thread_id):
        response = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id
        )
        return response

    def check_run_status(self, thread_id, run_id):
        response = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        return response

    def get_assistant_responses(self, thread_id):
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        return [msg for msg in messages.data if msg.role == 'assistant']
