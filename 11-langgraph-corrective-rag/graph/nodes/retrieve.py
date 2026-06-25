from typing import Any

from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> dict[str, Any]:
    print("--- Retrieve ---")
    question = state.get("question", "")

    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}
