import asyncio
from typing import Any
from graph.chains.retrieval_grade import retrieval_grader
from graph.state import GraphState

async def grade_documents(state: GraphState) -> dict[str, Any]:
    print("--- Check document relevance to the question ---")
    question = state["question"]
    documents = state["documents"]

    sem = asyncio.Semaphore(3)
    async def grade_with_limit(d):
        async with sem:
            print("""--- Checking one document ---""")

            return await retrieval_grader.ainvoke({"question": question, "document": d})

    coroutines = [grade_with_limit(d) for d in documents]
    scores = await asyncio.gather(*coroutines)

    filtered_docs = []
    web_search = False
    for doc, score in zip(documents, scores):

        grade = score.binary_score
        if grade.lower() == "yes":
            print("""--- GRADE: Document relevant ---""")
            filtered_docs.append(doc)
        else:
            print("""--- GRADE: Document not relevant ---""")

    if len(filtered_docs) == 0:
        web_search = True

    return {"documents": filtered_docs, "web_search": web_search, "question": question}
