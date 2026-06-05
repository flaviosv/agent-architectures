import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv(Path(__file__).parent / "../.env")

QUERY = "Search for 3 jobs posting exclusively in linkedin for ai engineer using langchain in the in the bay area and list their details"

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answers and sources"""
    answer: str = Field(description="The answer to the query")
    sources: list[Source] = Field(description="The sources used to answer the query", default_factory=list)

tavily = TavilyClient()

@tool
def search(query: str) -> dict:
    """
    Tool that searches over internet
    Args
        query: The query to search for
    Returns
        The search result
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)

tools = [search]


def run_groq_strategy() -> AgentResponse:
    """Groq doesn't support tools + structured output simultaneously.
    Workaround: agent runs with tools first, then a separate call structures the final answer."""
    llm = ChatGroq(temperature=0, model=os.getenv("GROQ_MODEL", ""))
    agent = create_agent(llm, tools)
    structured_llm = llm.with_structured_output(AgentResponse)
    result = agent.invoke({"messages": [HumanMessage(content=QUERY)]}, config={"run_name": "02-base-groq"})
    return structured_llm.invoke(result["messages"][-1].content)


def run_ollama_strategy() -> AgentResponse:
    """Ollama: ProviderStrategy enforces JSON output at provider level, guaranteeing structured_response.
    NOTE: costs an extra LLM call (full conversation + schema) to produce the structured output.
    response_format=AgentResponse directly (ToolStrategy) would skip that call but is unreliable
    after multiple tool calls with the available local models."""
    llm = ChatOllama(temperature=0, model=os.getenv("OLLAMA_MODEL", ""))
    agent = create_agent(llm, tools, response_format=ProviderStrategy(AgentResponse))
    result = agent.invoke({"messages": [HumanMessage(content=QUERY)]}, config={"run_name": "02-base-ollama"})
    return result["structured_response"]


def main():
    print("--- Groq Strategy ---")
    print(run_groq_strategy())

    print("--- Ollama Strategy ---")
    print(run_ollama_strategy())

if __name__ == "__main__":
    main()
