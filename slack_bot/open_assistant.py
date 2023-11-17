from time import sleep
from typing import Literal, Union
from openai import OpenAI
import logging
from slack_bot.exceptions import NoThreadCreatedError, NoAssistantCreatedError

client = OpenAI()

class OpenAssistant:
    def __init__(self, assistant_name: str = 'assistant', model: str = 'gpt-3.5-turbo', tools=None):
        self.assistant_name = assistant_name
        self.model = model
        self.tools = tools if tools else []
        self.assistant = self.create_assistant()
        self.thread = None
        self.message = None
        self.run = None

    def create_assistant(self):
        assistant = client.beta.assistants.create(
            name=self.assistant_name,
            model=self.model,
            tools=self.tools
        )
        return assistant

    def create_thread(self):
        self.thread = client.beta.threads.create()

        return self.thread

    def add_message_to_thread(self, content, role: Literal["user"], thread_id=None) -> None:
        if thread_id is None and self.thread:
            thread_id = self.thread.id
        else:
            raise NoThreadCreatedError()
        
        if content is None:
            logging.error('content cannot be None.')
            raise ValueError('content cannot be None.')
        
        try:
            self.message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content
            )
        except Exception as e:
            logging.error(e, exc_info=True)
    
    def create_run(self, thread_id: Union[str, None] = None, assistant_id: Union[str, None] = None) -> None:
        if thread_id is None and self.thread:
            thread_id = self.thread.id
        else:
            raise NoThreadCreatedError()
        
        if assistant_id is None and self.assistant:
            assistant_id = self.assistant.id
        else:
            raise NoAssistantCreatedError()
        
        self.run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )

    def check_run_status(self, thread_id, run_id):
        response = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        return response

    def get_assistant_responses(self, thread_id):
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        return [msg for msg in messages.data if msg.role == 'assistant']
    
    def provide_response(self, run_id, thread_id):
        for _ in range(10):
            sleep(3)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread_id)
                return [msg for msg in messages.data if msg.role == 'assistant'][0].content
            else:
                print('Run not completed yet. Waiting 3 seconds...')
