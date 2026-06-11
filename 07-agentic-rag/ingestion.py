import asyncio
import os
import ssl
import certifi
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_classic import text_splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import log_header, log_info, Colors, log_success, log_error
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

load_dotenv()

EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDING", "")
MODEL = os.getenv("OLLAMA_MODEL", "")

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

vector_store = PineconeVectorStore(
    index_name="langraph-udemy-eden-marco-rag-extra-concepts",
    embedding=embeddings
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=15, limit=50, max_pages=100)
tavily_crawl = TavilyCrawl(max_depth=5)

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    log_header("VECTOR STORAGE PHASE")
    log_info(f"VectorStore Indexing: Preparing to add {len(documents)} documents to vector store", Colors.DARKCYAN)

    batches = [documents[i: i + batch_size] for i in range(0, len(documents), batch_size)]

    log_info(f"VectorStore indexing: Split into {len(batches)} batches of {batch_size} documents each")

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vector_store.aadd_documents(batch)
            log_success(f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} documents")
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num}/{len(batches)} documents: {str(e)}")

            return False

        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]

    results = await asyncio.gather(*tasks)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(f"VectorStore Indexing: Successfully added all {len(batches)} batches")
    else:
        log_error(f"VectorStore Indexing: Failed to add {len(batches) - successful} batches")

async def main():
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("TavilyCrawl: Stargint to crawl documentatuon from https://python.langchain.com/", Colors.PURPLE)

    res = tavily_crawl.invoke({
        "url": "https://docs.langchain.com/oss/python/langchain/overview",
        "extract_depth": "advanced"
    })

    all_docs = [Document(page_content=result['raw_content'] if result['raw_content'] else "No content", metadata={"source": result["url"]}) for result in res["results"]]

    log_success(f"TavilyCrawl: Finished crawling documentation from https://python.langchain.com/, found {len(all_docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)

    log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents")

    await index_documents_async(splitted_docs, batch_size=500)

if __name__ == "__main__":
    asyncio.run(main())
