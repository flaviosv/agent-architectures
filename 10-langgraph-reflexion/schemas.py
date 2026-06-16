from pydantic import BaseModel, Field

class AnswerQuestion(BaseModel):
    """Answer the question."""
    answer: str = Field(description="~250 word detailed answer to the question")
    missing: str = Field(description="Critique of what is missing from your answer")
    superfluous: str = Field(description="Critique of what is superfluous in your answer")
    search_queries: list[str] = Field(description="1-3 search queries for researching improvements to address the critique of your answer")

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""

    references: list[str] = Field(description="Citations motivating your updated answer")
