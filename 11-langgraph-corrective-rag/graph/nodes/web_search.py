from dotenv import load_dotenv
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

load_dotenv(Path(__file__).parent / "../../../.env")

web_search_tool = TavilySearch(max_results=1)

def web_search(state: GraphState) -> dict[str, Any]:
    print("--- WEB SEARCH ---")
    question = state.get("question", "")
    documents = state.get("documents", "")

    results = web_search_tool.invoke({"query": question})
    results = "\n".join(results)

    document = Document(page_content=results)
    if documents is not None:
        documents.append(results)
    else:
        documents = [results]

        return {"documents": documents, "question": question}
