# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

"""
DESCRIPTION:
    This sample demonstrates how to use basic agent operations for vector search on your 
    Postgres database and the Azure Agents service. The sample uses a PostgreSQL database
    to fetch cases information based on a query and returns the information as a JSON string.
    The sample also demonstrates how to use the Azure Agents service to create an agent,
    thread, message, and run to interact with the user.
    
    Also you will be using a synchronous client with Azure Monitor tracing.
    View the results in the "Tracing" tab in your Azure AI Foundry project page.

USAGE:
    python agents_with_function_calling_with postgres.py

    Before running the sample:

    pip install -r requirements.txt

    Set these environment variables with your own values:
    1) PROJECT_CONNECTION_STRING - The project connection string, as found in the overview page of your
       Azure AI Foundry project.
    2) MODEL_DEPLOYMENT_NAME - The deployment name of the AI model, as found under the "Name" column in 
       the "Models + endpoints" tab in your Azure AI Foundry project.
    3) AZURE_PG_CONNECTION - The connection string for your PostgreSQL database.
    4) EMBEDDING_MODEL_NAME - The deployment name of the AI embedding model used for vector search, as found under the "Name" column in 
       the "Models + endpoints" tab in your Azure AI Foundry project.
    5) AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED - Optional. Set to `true` to trace the content of chat
       messages, which may contain personal data. False by default.
"""
from typing import Any, Callable, Set

import os, time, json
from datetime import datetime
from azure.ai.projects import AIProjectClient
from azure.ai.projects.telemetry import trace_function
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import FunctionTool, RequiredFunctionToolCall, SubmitToolOutputsAction, ToolOutput, ToolSet
from opentelemetry import trace
from azure.monitor.opentelemetry import configure_azure_monitor
from pprint import pprint
from legal_agent_tools import user_functions
from dotenv import load_dotenv
# Load environment variables
load_dotenv("../.env")

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(), 
    conn_str=os.environ["PROJECT_CONNECTION_STRING"]
)

# Enable Azure Monitor tracing
application_insights_connection_string = project_client.telemetry.get_connection_string()
if not application_insights_connection_string:
    print("Application Insights was not enabled for this project.")
    print("Enable it via the 'Tracing' tab in your AI Foundry project page.")
    exit()
configure_azure_monitor(connection_string=application_insights_connection_string)

project_client.telemetry.enable() # Enable Azure Monitor tracing

scenario = os.path.basename(__file__)
tracer = trace.get_tracer(__name__)

# Initialize agent toolset with user functions
functions = FunctionTool(user_functions)
toolset = ToolSet()
toolset.add(functions)

with tracer.start_as_current_span(scenario):
    with project_client:
        # Create an agent and run user's request with function calls
        agent = project_client.agents.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name=f"advanced-documents-agent-{datetime.now().strftime('%Y%m%d%H%M')}",
            instructions=f"""
            You are a helpful assistant that can retrieve information from various types of documents. 
            The current date is {datetime.now().strftime('%Y-%m-%d')}.
            """,
            toolset=toolset,
        )
        print(f"Created agent, ID: {agent.id}")

        thread = project_client.agents.create_thread()
        print(f"Created thread, ID: {thread.id}")

        message = project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content="What are shellfish?",
        )
        print(f"Created message, ID: {message.id}")

        run = project_client.agents.create_run(thread_id=thread.id, agent_id=agent.id)
        print(f"Created run, ID: {run.id}")

        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(1)
            run = project_client.agents.get_run(thread_id=thread.id, run_id=run.id)
            print(f"Current run status: {run.status}")

            if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                print(f"Tool calls: {tool_calls}")
                if not tool_calls:
                    print("No tool calls provided - cancelling run")
                    project_client.agents.cancel_run(thread_id=thread.id, run_id=run.id)
                    break

                tool_outputs = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, RequiredFunctionToolCall):
                        try:
                            output = functions.execute(tool_call)
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
                    project_client.agents.submit_tool_outputs_to_run(
                        thread_id=thread.id, 
                        run_id=run.id, 
                        tool_outputs=tool_outputs
                    )

            print(f"Current run status: {run.status}")

        print(f"Run completed with status: {run.status}")

        # Delete the agent when done
        # project_client.agents.delete_agent(agent.id)
        # print("Deleted agent")

        # Fetch and log all messages
        messages = project_client.agents.list_messages(thread_id=thread.id)
        print(f"Messages: {messages}")
        pprint(messages['data'][0]['content'][0]['text']['value'])