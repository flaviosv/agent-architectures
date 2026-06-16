import asyncio
from pathlib import Path

from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from chains import generate_chain, reflect_chain

load_dotenv(Path(__file__).parent / "../.env")

REFLECT = "reflect"
GENERATE = "generate"

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def generation_node(state: MessageGraph):
    res = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [res]}

"""
The reflection returns a HumanMessage, the idea is to make the LLM understand that an human generated this reflecton
This is a prompt technique to make the LLM chat about the critique getting from a human behavior
"""
def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

def should_continue(state: MessageGraph) -> str:
    if len(state["messages"]) > 6:
        return END

    return REFLECT

async def main(prompt: str):
    builder = StateGraph(state_schema=MessageGraph)
    builder.add_node(GENERATE, generation_node)
    builder.add_node(REFLECT, reflection_node)
    builder.set_entry_point(GENERATE)

    builder.add_conditional_edges(GENERATE, should_continue, {
        END: END,
        REFLECT: REFLECT
    })
    builder.add_edge(REFLECT, GENERATE)

    compiled = builder.compile()

    res = await compiled.ainvoke({"messages": [HumanMessage(content=prompt)]})
    compiled.get_graph().draw_mermaid_png(output_file_path="flow.png")
    # print(compiled.get_graph().print_ascii())

    print(res["messages"][-1].content)


if __name__ == "__main__":
    query = """
The outrageous effectiveness of Leitwörter

I've realised that all of the great skills I've written share one thing in common.

They make heavy use of Leitwörter - leading words.

A leitwort comes from literary theory. It's a repeated word or phrase used throughout a text to establish a theme or anchor meaning.

In skills, a leitwort is a word or phrase the agent uses to guide its own behavior. In other words, it's a word that leads the agent in a certain direction.

Let's take the leitwort "zone of proximal development" from my /teach skill. It's a phrase from the study of education. It means the "zone where the user feels challenged but not overwhelmed".

I use this only a couple of times throughout the skill's SKILL.md, but I've seen it almost every time the agent invokes the skill.

- "Let me adjust the lesson so it's in the user's zone of proximal development."
- "I'll read the learning records to establish the user's zone of proximal development."

In other words, that single phrase encodes how the agent should behave, in a concise token the agent can itself repeat to reinforce its own behavior.

Not only that, but it also likely tickles the agents' parameters related to educational research and "being a good teacher".

For engineering, leading words like "tracer bullets", "deep modules", "test seams", "clean code" are outrageously effective for leading the agent to produce better code.

So a leitwort in AI is any word or phrase you use that appears in the agents' thinking traces and guides its behavior.

Enjoy finding your own.
    """
    res = asyncio.run(main(query))
