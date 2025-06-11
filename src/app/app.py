from flask import Flask, request, jsonify, render_template, session, g
from flask_session import Session
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    AgentEventHandler,
    FunctionTool,
    ListSortOrder,
    ToolOutput,
    RequiredFunctionToolCall,
    SubmitToolOutputsAction,
    MessageRole,
    RunStep,
    ThreadMessage,
    ThreadRun,
)
from dotenv import load_dotenv
from datetime import datetime
import os
import sys
import logging
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from legal_agent_tools import user_functions  # or import your function set as needed

# Configure logging to output to Azure log stream (stdout)
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Load environment variables from .env file
load_dotenv("../../.env")

# Initialize Flask application
app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Initialize Azure AgentsClient
agents_client = AgentsClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True,
        exclude_shared_token_cache_credential=True,
    ),
)

# Define user functions as a set for FunctionTool
functions = FunctionTool(functions=user_functions)

# Create an agent with a specific model and instructions
agent = agents_client.create_agent(
    model=os.environ["MODEL_DEPLOYMENT_NAME"],
    name=f"chat-agent-{datetime.now().strftime('%Y%m%d%H%M')}",
    instructions=(
            "You are a helpful assistant that can retrieve information from a dataset of invoices.\n"
            "For every user question, you MUST use the sql_search tool to answer. Do not answer from your own knowledge, search the internet, or make up data.\n"
            f"The current date is {datetime.now().strftime('%Y-%m-%d')}."
    ),
    tools=functions.definitions,
)
logger.debug(f"Agent created with ID: {agent.id}")

# Function to create a thread for the user session
def create_thread():
    thread = agents_client.threads.create()
    session["thread_id"] = thread.id
    logger.info(f"Created thread with ID: {session['thread_id']}")
    return session["thread_id"]

# Implement the event handler class
class MyEventHandler(AgentEventHandler):
    def __init__(self, functions: FunctionTool) -> None:
        super().__init__()
        self.functions = functions

    def on_thread_message(self, message: "ThreadMessage") -> None:
        print(f"ThreadMessage created. ID: {message.id}, Status: {message.status}")

    def on_thread_run(self, run: "ThreadRun") -> None:
        print(f"ThreadRun status: {run.status}")

        if run.status == "failed":
            print(f"Run failed. Error: {run.last_error}")

        if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                if isinstance(tool_call, RequiredFunctionToolCall):
                    try:
                        output = self.functions.execute(tool_call)
                        tool_outputs.append(
                            ToolOutput(
                                tool_call_id=tool_call.id,
                                output=output,
                            )
                        )
                    except Exception as e:
                        print(f"Error executing tool_call {tool_call.id}: {e}")

            print(f"Tool outputs: {tool_outputs}")
            if tool_outputs:
                agents_client.runs.submit_tool_outputs_stream(
                    thread_id=run.thread_id, run_id=run.id, tool_outputs=tool_outputs, event_handler=self
                )

    def on_run_step(self, step: "RunStep") -> None:
        print(f"RunStep type: {step.type}, Status: {step.status}")

    def on_error(self, data: str) -> None:
        print(f"An error occurred. Data: {data}")

    def on_done(self) -> None:
        print("Stream completed.")

    def on_unhandled_event(self, event_type: str, event_data: any) -> None:
        print(f"Unhandled Event Type: {event_type}, Data: {event_data}")

# Define the root route to render the index.html page
@app.route("/")
def index():
    try:
        # Create a new thread for the user session
        thread_id = create_thread()
        logger.debug(f"New thread ID for session: {thread_id}")
    except Exception as e:
        logger.exception(f"Error creating thread on index load: {e}")
    return render_template("index.html")

def format_agent_response(text):
    # Convert **bold** to <strong>
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Replace numbered lists with <ol><li>...</li></ol>
    def list_replacer(match):
        items = re.findall(r"\d+\.\s+(.+?)(?=(?:\d+\.|$))", match.group(0), re.DOTALL)
        return "<ol>" + "".join(f"<li>{item.strip()}</li>" for item in items) + "</ol>"
    text = re.sub(r"((?:\d+\..+?)+)", list_replacer, text, flags=re.DOTALL)
    # Replace newlines with <br>
    text = text.replace("\n", "<br>")
    return text

# Define the /chat endpoint to handle user messages
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Retrieve the user message from the request
        user_message = request.json.get("message")
        logger.debug(f"Received message: {user_message}")

        if not user_message:
            logger.warning("No message provided in the request.")
            return jsonify({"error": "Message is required"}), 400

        # Use the thread created for the current session
        thread_id = session.get("thread_id")
        if not thread_id:
            logger.warning("No thread ID found in session. Creating a new thread.")
            thread_id = create_thread()
            logger.debug(f"New thread ID created: {thread_id}")

        # Add the user message to the thread
        agents_client.messages.create(
            thread_id=thread_id,
            role=MessageRole.USER,
            content=user_message,
        )
        logger.debug("User message added to thread.")

        # Process the thread using the agent with streaming and event handler
        with agents_client.runs.stream(
            thread_id=thread_id,
            agent_id=agent.id,
            event_handler=MyEventHandler(functions)
        ) as stream:
            stream.until_done()

        # Retrieve the agent's response from the thread
        messages = agents_client.messages.list(thread_id=thread_id, order=ListSortOrder.ASCENDING)
        agent_response = None
        for msg in messages:
            if msg.role == MessageRole.AGENT and msg.text_messages:
                agent_response = msg.text_messages[-1].text.value
        if not agent_response:
            agent_response = "No response from the agent."
            logger.warning("No assistant response found in messages.")

        logger.info(f"Agent response: {agent_response}")

        # Format the response as HTML
        formatted_response = format_agent_response(agent_response)

        return jsonify({"response": formatted_response})
    except Exception as e:
        logger.exception(f"Error in /chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Define the /delete_thread endpoint to handle thread deletion
@app.route("/delete_thread", methods=["POST"])
def delete_thread():
    try:
        if "thread_id" in session:
            thread_id = session["thread_id"]
            agents_client.threads.delete(thread_id=thread_id)
            logger.info(f"Deleted thread with ID: {thread_id}")
            session.pop("thread_id", None)  # Remove thread_id from session
        return jsonify({"message": "Thread deleted successfully"}), 200
    except Exception as e:
        logger.exception(f"Error in /delete_thread endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Start the Flask application
if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000)
