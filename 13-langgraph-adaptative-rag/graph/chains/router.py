import os
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

class RouteQuery(BaseModel):

    datasource: Literal["vectorstore", "websearch"] = Field(..., description="Given a user question choose to route it to websearch or a vectorstore")

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", ""))
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """
You are an expert at routing a user question to a vectorstore or websearch
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system),
        HumanMessage(content="{question}")
    ]
)

question_router = route_prompt | structured_llm_router
