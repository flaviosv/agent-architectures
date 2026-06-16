import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

load_dotenv(Path(__file__).parent / "../.env")

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet. "
        "Always provide detailed recommendations, including requests for length, virality, style, etc."
    ),
    MessagesPlaceholder(variable_name="messages")
])

generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a twitter techie influencer assistant tasked with writing excellent twitter posts. "
        "Generate the best twitter possible for the user's request. "
        "If the user provides critique, respond with a revised version of your previous attempts. "
    ),
    MessagesPlaceholder(variable_name="messages")
])

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"))
reflect_chain = reflection_prompt | llm
generate_chain = generation_prompt | llm
