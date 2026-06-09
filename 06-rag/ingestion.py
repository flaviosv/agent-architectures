import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_classic import text_splitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(Path(__file__).resolve().parent / "mediumblog1.txt")
    documents = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=8)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING", ""))

    print("pineconing...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name="langraph-udemy-eden-marco-rag")
    print("finished")


