from flask import Flask, request, jsonify, render_template, session, g
from flask_session import Session
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import FunctionTool, ToolSet
from dotenv import load_dotenv
from datetime import datetime
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from legal_agent_tools import user_functions

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

# Configure Flask session
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Initialize Azure AI Client and agent
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)

# Create a toolset with user-defined functions
functions = FunctionTool(user_functions)  # Ensure this references the correct set
toolset = ToolSet()
toolset.add(functions)
project_client.agents._function_tool = functions # Include this line if seeing "function not found" error

# Create an agent with a specific model and instructions
agent = project_client.agents.create_agent(
    model=os.environ["MODEL_DEPLOYMENT_NAME"],
    name=f"chat-agent-{datetime.now().strftime('%Y%m%d%H%M')}",
    instructions=f"""
    You are a helpful assistant that retrieves information from documents and can answer questions based on that information. Always search through the documents and retrieve information.
    If no relevant documents were retrieved, do not answer the question.
    The current date is {datetime.now().strftime('%Y-%m-%d')}.
    """,
    toolset=toolset,
)

logger.debug(f"Agent created with ID: {agent.id}")

# Function to create a thread for the user session
def create_thread():
    # Always create a new thread for the user session
    thread = project_client.agents.create_thread()
    session["thread_id"] = thread.id
    logger.info(f"Created thread with ID: {session['thread_id']}")
    return session["thread_id"]

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
        project_client.agents.create_message(
            thread_id=thread_id,
            role="user",
            content=user_message,
        )
        logger.debug("User message added to thread.")

        # Process the thread using the agent
        run = project_client.agents.create_and_process_run(thread_id=thread_id, agent_id=agent.id)
        logger.debug(f"Run status: {run.status}")

        if run.status == "failed":
            logger.error(f"Run failed with error: {run.last_error}")
            # Extract error details from RunError
            error_details = {
                "code": getattr(run.last_error, "code", "unknown_error"),
                "message": getattr(run.last_error, "message", "An unknown error occurred."),
            }
            return jsonify({"error": error_details}), 500

        # Retrieve the agent's response from the thread
        messages = project_client.agents.list_messages(thread_id=thread_id)
        logger.debug(f"Retrieved messages: {messages}")

        # Find the assistant's response (assume it's the last message)
        agent_response = None
        for message in messages["data"]:
            if message["role"] == "assistant":
                agent_response = message["content"][0]["text"]["value"]
                logger.debug(f"Agent response: {agent_response}")
                break

        if not agent_response:
            agent_response = "No response from the agent."
            logger.warning("No assistant response found in messages.")

        return jsonify({"response": agent_response})
    except Exception as e:
        logger.exception(f"Error in /chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Define the /delete_thread endpoint to handle thread deletion
@app.route("/delete_thread", methods=["POST"])
def delete_thread():
    try:
        if "thread_id" in session:
            thread_id = session["thread_id"]
            project_client.agents.delete_thread(thread_id=thread_id)
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
