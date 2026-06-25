import os
from dotenv import load_dotenv
from pathlib import Path
from langsmith.client import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

load_dotenv(Path(__file__).parent / "../../../.env")

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", ""))
client = Client()
prompt = client.pull_prompt(prompt_identifier="rlm/rag-prompt", dangerously_pull_public_prompt=True)

generation_chain = prompt | llm | StrOutputParser()
