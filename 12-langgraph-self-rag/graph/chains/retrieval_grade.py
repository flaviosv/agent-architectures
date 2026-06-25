import os

from dotenv import load_dotenv
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

load_dotenv(Path(__file__).parent / "../../../.env")

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", ""))

class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system =  """\
/no_think
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant
Give a binary score, 'yes' or 'no' score to indicate whether the document is relevante to the question
"""

human = """\
Retrieved document: {document}
User Question: {question}
"""

grade_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system),
    HumanMessage(content=human),
])

retrieval_grader = grade_prompt | structured_llm_grader
