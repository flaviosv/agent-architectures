from dotenv import load_dotenv
from pathlib import Path

from graph.chains.retrieval_grade import GradeDocuments, retrieval_grader

load_dotenv(Path(__file__).parent / "../../../../.env")

def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeDocuments = retrieval_grader.invoke({
        "question": question,
        "document": "\n".join([doc.page_content for doc in docs])
    })

    assert res.binary_score == "yes"
