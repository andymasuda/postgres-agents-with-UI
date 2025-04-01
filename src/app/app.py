from flask import Flask, request, jsonify, render_template
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import FunctionTool, ToolSet
from dotenv import load_dotenv
from datetime import datetime
import os
import sys
import logging
import atexit

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

# Initialize Azure AI Client and agent
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)

# Create a toolset with user-defined functions
functions = FunctionTool(user_functions)
toolset = ToolSet()
toolset.add(functions)

# Create an agent with a specific model and instructions
agent = project_client.agents.create_agent(
    model=os.environ["MODEL_DEPLOYMENT_NAME"],
    name=f"chat-agent-{datetime.now().strftime('%Y%m%d%H%M')}",
    instructions=f"""
    You are a helpful assistant that can retrieve information from various types of documents and answer related questions/requests. 
    The current date is {datetime.now().strftime('%Y-%m-%d')}.
    """,
    toolset=toolset,
)

# List to keep track of created thread IDs
created_threads = []

# Function to delete the agent and threads when the app is closed
def cleanup_agent_and_threads():
    try:
        # Delete all created threads
        for thread_id in created_threads:
            try:
                logger.info(f"Deleting thread with ID: {thread_id}")
                project_client.agents.delete_thread(thread_id=thread_id)
                logger.info(f"Thread {thread_id} deleted successfully.")
            except Exception as e:
                logger.error(f"Failed to delete thread {thread_id}: {e}")

        # Delete the agent
        logger.info(f"Deleting agent with ID: {agent.id}")
        project_client.agents.delete_agent(agent_id=agent.id)
        logger.info("Agent deleted successfully.")
    except Exception as e:
        logger.error(f"Failed to clean up resources: {e}")

# Register the cleanup function to run at exit
atexit.register(cleanup_agent_and_threads)

# Define the root route to render the index.html page
@app.route("/")
def index():
    logger.debug("Rendering index.html")
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

        # Create a new thread for the conversation
        thread = project_client.agents.create_thread()
        logger.debug(f"Created thread with ID: {thread.id}")

        # Track the created thread ID
        created_threads.append(thread.id)

        # Add the user message to the thread
        project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=user_message,
        )
        logger.debug("User message added to thread.")

        # Process the thread using the agent
        run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
        logger.debug(f"Run status: {run.status}")

        if run.status == "failed":
            logger.error(f"Run failed with error: {run.last_error}")
            return jsonify({"error": run.last_error}), 500

        # Retrieve the agent's response from the thread
        messages = project_client.agents.list_messages(thread_id=thread.id)
        logger.debug(f"Retrieved messages: {messages}")

        for message in reversed(messages["data"]):
            if message["role"] == "assistant":
                agent_response = message["content"][0]["text"]["value"]
                logger.debug(f"Agent response: {agent_response}")
                break
        else:
            agent_response = "No response from the agent."
            logger.warning("No assistant response found in messages.")

        return jsonify({"response": agent_response})
    except Exception as e:
        logger.exception(f"Error in /chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Start the Flask application
if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000)
