import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv(Path(__file__).parent / "../.env")

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", ""))
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """\
You are an expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe yo maximize improvement.
3. Recommend search queries to research information and improve your answer.
"""
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        "Answer the user's questions above using the required format."
    )
]).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~25- word answer."
)
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """\
Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not count towards the word limitation)
        - [1] https://example.com
        - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your anser and make SURE is notmore than 250 characters
"""

"""
When sending a pydantic object through the tools and tool_choice, internally LangChain converts pydantic in JSON schema
tool_choice make sure the LLM will call this tool mandatory
"""
revisor = (
        actor_prompt_template.partial(first_instruction=revise_instructions)
        | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer"
))
