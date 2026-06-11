import os
from typing import Any, Dict
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage, HumanMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv(Path(__file__).parent / "../../.env")

EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDING", "")
MODEL = os.getenv("OLLAMA_MODEL", "")

embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

vector_store = PineconeVectorStore(
    index_name="langraph-udemy-eden-marco-rag-extra-concepts",
    embedding=embeddings
)

model = init_chat_model(model=f"ollama:{MODEL}", temperature=0)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str) :
    """Retrieve relevant documentat to help answer user queries about LangChain"""
    retrieved_docs = vector_store.as_retriever().invoke(query, k=4)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', "Unknown")}\n\nContent: {doc.page_content}" for doc in retrieved_docs
    )

    return serialized, retrieved_docs

def main(query: str) -> Dict[str, Any]:
    system_prompt = (
        "You are a helpful AI Assistant that answers questions about LangChain documentation"
        "You have access to a tool that retrieves relevant documentation"
        "Use the tool to find relevant information before answering questions"
        "Always cite the sources you use in your answers"
        "If you can't find the answer in the retrieved documentation, say so"
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    messages = [HumanMessage(content=query)]

    response = agent.invoke({"messages": messages})

    answer = response["messages"][-1].content

    context_docs = (
        [message.artifact
         for message in response["messages"]
         if isinstance(message, ToolMessage) and hasattr(message, "artifact") and isinstance(message.artifact, list)
        ]
    )

    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == "__main__":
    result = main("What are deep agents?")
    print(result)
