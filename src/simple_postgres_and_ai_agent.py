import os
import sys
import time
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    AgentEventHandler,
    FunctionTool,
    ListSortOrder,
    MessageDeltaChunk,
    RequiredFunctionToolCall,
    RunStep,
    SubmitToolOutputsAction,
    ThreadMessage,
    ThreadRun,
    ToolOutput,
    MessageRole,
)
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from legal_agent_tools import sql_search, vector_search

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

agents_client = AgentsClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True,
        exclude_shared_token_cache_credential=True,
    ),
)

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

with agents_client:
    # Define user functions as a set for FunctionTool
    user_functions = {sql_search, vector_search}
    functions = FunctionTool(functions=user_functions)

    agent = agents_client.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name=f"invoice-agent-{int(time.time())}",
        instructions=(
            "You are an intelligent routing assistant for a database of invoices, designed for internal business analysts. "
            "Your primary task is to analyze the user's question and decide which of the two available tools is the most appropriate to answer it. "
            "You must respond by calling one of the tools. Do not answer the question directly from your own knowledge.\n\n"
            
            "**TOOL DEFINITIONS:**\n\n"
            
            "1. `sql_search(query: str)`:\n"
            "   - **Description:** This is your default, primary tool. Use it for the vast majority of questions. It is extremely powerful for precise queries involving specific keywords, filtering on known categories (like region or product type), date ranges, and performing calculations (like SUM, COUNT, AVG).\n"
            "   - **Use For Examples:**\n"
            "     - 'What is the total profit for invoices for ACME Corp?'\n"
            "     - 'Show me invoices for ACME Corp in the Central region.'\n"
            "     - 'How many sales did we have last week?'\n"
            "     - 'What are the transactions involving specialty lumber or engineered wood?'\n"
            
            "2. `vector_search(query: str)`:\n"
            "   - **Description:** This is a specialized tool for conceptual or similarity-based analysis where specific keywords are not enough.\n"
            "   - **Use For Examples:**\n"
            "     - 'Which invoices are similar to invoice #9999?'\n"
            "     - 'Analyze sales patterns for our premium-grade materials.'\n\n"
            
            "**DECISION-MAKING PROCESS:**\n"
            "1. **Default to `sql_search`**. It can handle almost any question about filtering, counting, and summarizing data.\n"
            "2. **Only choose `vector_search`** when the user's query is not about finding specific keywords, but about performing one of these three analytical tasks:\n"
            "   - **Similarity Analysis:** For finding comparable transactions or items based on an example. (e.g., 'Find deals similar to invoice #9876', 'Show me alternatives to this product we sold').\n"
            "   - **Identifying Purchase Patterns:** To find invoices that represent a certain type of project or complete customer order. (e.g., 'Show me sales that look like a residential roofing project').\n"
            "   - **Searching by Abstract Business Concepts:** To find sales based on strategic ideas or qualities not in the data. (e.g., 'What were our sales for contractor-grade materials?', 'Analyze our high-margin product sales').\n\n"
            
            f"For any queries involving dates, remember that the current date is {datetime.now().strftime('%Y-%m-%d')}."
        ),
        tools=functions.definitions,
    )
    print(f"Created agent, agent ID: {agent.id}")

    thread = agents_client.threads.create()
    print(f"Created thread, thread ID {thread.id}")

    message = agents_client.messages.create(
        thread_id=thread.id,
        role="user",
        content="Analyze sales patterns for our premium-grade materials.",
    )
    print(f"Created message, message ID {message.id}")

    print("[DEBUG] About to enter run stream context")
    with agents_client.runs.stream(
        thread_id=thread.id, agent_id=agent.id, event_handler=MyEventHandler(functions)
    ) as stream:
        print("[DEBUG] Entered run stream context, about to call until_done()")
        stream.until_done()
        print("[DEBUG] Finished stream.until_done()")
    print("[DEBUG] Exited run stream context")

    agents_client.delete_agent(agent.id)
    print("Deleted agent")

    messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
    for msg in messages:
        if msg.text_messages:
            last_text = msg.text_messages[-1]
            print(f"{msg.role}: {last_text.text.value}")

