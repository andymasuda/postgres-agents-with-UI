from flask import Flask, request, jsonify, render_template
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import FunctionTool, ToolSet
from dotenv import load_dotenv
from datetime import datetime
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from legal_agent_tools import user_functions

# Load environment variables
load_dotenv("../../.env")

app = Flask(__name__)

# Initialize Azure AI Client and agent
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)

functions = FunctionTool(user_functions)
toolset = ToolSet()
toolset.add(functions)

agent = project_client.agents.create_agent(
    model=os.environ["MODEL_DEPLOYMENT_NAME"],
    name=f"chat-agent-{datetime.now().strftime('%Y%m%d%H%M')}",
    instructions=f"""
    You are a helpful assistant that can retrieve information from various types of documents. 
    The current date is {datetime.now().strftime('%Y-%m-%d')}.
    """,
    toolset=toolset,
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    thread = project_client.agents.create_thread()
    project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)

    if run.status == "failed":
        return jsonify({"error": run.last_error}), 500

    # Retrieve the agent's response from the last message in the thread
    messages = project_client.agents.list_messages(thread_id=thread.id)
    for message in reversed(messages["data"]):
        if message["role"] == "assistant":
            agent_response = message["content"][0]["text"]["value"]
            break
    else:
        agent_response = "No response from the agent."

    return jsonify({"response": agent_response})

if __name__ == "__main__":
    app.run(debug=True)
