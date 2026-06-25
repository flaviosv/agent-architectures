import os

from dotenv import load_dotenv
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

load_dotenv(Path(__file__).parent / "../../../.env")

llm =ChatOllama(model=os.getenv("OLLAMA_MODEL", ""))

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer address the question, 'yes' or 'no'")


structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """\
You are a grader assessing whether an answer address / resolves a question
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question
"""

answer_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system),
    HumanMessage(content="Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
])

answer_grader = answer_prompt | structured_llm_grader
