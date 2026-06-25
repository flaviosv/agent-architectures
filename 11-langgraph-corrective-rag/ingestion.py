import os

import chromadb
from dotenv import load_dotenv
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv(Path(__file__).parent / "../.env")

chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", ""),
    port=int(os.getenv("CHROMA_PORT", 0))
)

collection_name = "langraph-udemy-eden-marco-langgraph-advanced-rag"

print("--- Creating a retriever ---")
retriever = Chroma(
    collection_name=collection_name,
    embedding_function=OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING", "")),
    client=chroma_client
).as_retriever()

if __name__ == "__main__":
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    docs_splits = text_splitter.split_documents(docs_list)

    print("--- Resetting collection ---")
    Chroma(
        client=chroma_client,
        collection_name=collection_name
    ).reset_collection()

    print("--- Adding documents ---")
    vectorstore = Chroma.from_documents(
        documents=docs_splits,
        collection_name=collection_name,
        embedding=OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING", "")),
        client=chroma_client
    )
