from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain.messages import SystemMessage

from react import llm, tools

load_dotenv()

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
