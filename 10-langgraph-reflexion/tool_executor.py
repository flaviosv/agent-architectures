from pathlib import Path
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv(Path(__file__).parent / "../.env")

tavily_tool = TavilySearch(max_results=1)

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

"""
The difference of using StructureTools rather than @tool is that with from_function i can change the tool name
As the name will be i.e AnswerQuestion, matches the tool i'm adding to the LLM
"""
execute_tools = ToolNode([
    StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
    StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
])
