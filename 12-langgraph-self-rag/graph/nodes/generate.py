from typing import Any

from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> dict[str, Any]:
    print("--- GENERATE ---")
    question = state.get("question", "")
    documents = state.get("documents", [])

    generation = generation_chain.invoke({"context": documents, "question": question})

    return {"documents": documents, "generation": generation, "question": question}
