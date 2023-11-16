import openai

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import Assistant, Message, Run, Thread

class OpenAssistant:
    def __init__(self, api_key, assistant_name, model, tools=None):
        openai.api_key = api_key
        self.assistant_name = assistant_name
        self.model = model
        self.tools = tools if tools else []
        self.assistant_id = self.create_assistant()

    def create_assistant(self):
        response = openai.Assistant.create(
            name=self.assistant_name,
            model=self.model,
            tools=self.tools
        )
        return response['id']

    def create_thread(self):
        response = openai.Thread.create()
        return response['id']

    def add_message_to_thread(self, thread_id, message):
        response = openai.Message.create(
            thread_id=thread_id,
            role="user",
            content=message
        )
        return response

    def run_assistant(self, thread_id):
        response = openai.Run.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id
        )
        return response

    def check_run_status(self, thread_id, run_id):
        response = openai.Run.retrieve(thread_id=thread_id, run_id=run_id)
        return response

    def get_assistant_responses(self, thread_id):
        messages = openai.Message.list(thread_id=thread_id)
        return [msg for msg in messages['data'] if msg['role'] == 'assistant']
