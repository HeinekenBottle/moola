from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Upload CSV
file = client.files.create(file=open("data.csv", "rb"), purpose="assistants")
print(f"File ID: {file.id}")  # Use this in agent

# Create Assistant (your Agent A config)
assistant = client.beta.assistants.create(
    name="RWKV-TS Agent",
    instructions="Fit RWKV-TS on CSV file_id; output metrics.",
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}],
    tool_resources={"code_interpreter": {"file_ids": [file.id]}}
)

# Run Thread
thread = client.beta.threads.create()
client.beta.threads.messages.create(thread_id=thread.id, role="user", content="Train model")
run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)

# Poll & Get Output
while run.status != "completed": run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
messages = client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content[0].text.value)  # Text/JSON output