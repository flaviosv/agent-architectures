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

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """\
You are a grader assessing whether an LLM generation is grounded in / supported by a set retrieved facts
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts
"""

hallucination_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system),
    HumanMessage(content="Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
])

hallucination_grader = hallucination_prompt | structured_llm_grader
