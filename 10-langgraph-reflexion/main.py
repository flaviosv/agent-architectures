import asyncio
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_INTERACTIONS = 2

load_dotenv(Path(__file__).parent / "../.env")

def draft_node(state: MessagesState):
    print("Executing Draft Node")
    response = first_responder.invoke({"messages": state.get("messages")})

    return {"messages": [response]}

def revise_node(state: MessagesState):
    print("Executing Revise Node")
    response = revisor.invoke({"messages": state.get("messages")})

    return {"messages": [response]}

def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    print("Checking event loop")
    count_tools_visits = sum(isinstance(item, ToolMessage) for item in state.get("messages"))

    if count_tools_visits >= MAX_INTERACTIONS:
        return END

    return "execute_tools"


async def main(query: str):
    builder = StateGraph(MessagesState)
    builder.add_node("draft", draft_node)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("revise", revise_node)

    builder.add_edge(START, "draft")
    builder.add_edge("draft", "execute_tools")
    builder.add_edge("execute_tools", "revise")
    builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

    graph = builder.compile()
    graph.get_graph().draw_mermaid_png(output_file_path="flow.png")
    return await graph.ainvoke({"messages": [HumanMessage(content=query)]})

if __name__ == "__main__":
    query = "Create some sort of content about AI Reflexion Architecture"
    res = asyncio.run(main(query))

    print(res["messages"][-1].content)
