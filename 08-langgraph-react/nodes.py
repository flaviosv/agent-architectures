from dotenv import load_dotenv
from pathlib import Path
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain.messages import SystemMessage

from react import llm, tools

load_dotenv(Path(__file__).parent / "../.env")

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning mode
    """

    response = llm.invoke([SystemMessage(content=SYSTEM_MESSAGE), *state.get("messages", {})])

    return {"messages": [response]}

tool_node = ToolNode(tools)
