from openai import OpenAI
from dotenv import load_dotenv
import os
from time import sleep

load_dotenv()
client = OpenAI()

def provide_response(run_id, thread_id):
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


sql_dev = client.beta.assistants.retrieve(os.environ.get('ASSISTANT_ID') or '')
thread = client.beta.threads.create()
thread_message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="1 + 1 = ?",
)
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=sql_dev.id
)

response = provide_response(run.id, thread.id)
print(response)
