from dotenv import load_dotenv
from pathlib import Path
from langgraph.graph import END, StateGraph
from .consts import GRADE_DOCUMENTS, GENERATE, WEB_SEARCH, RETRIEVE
from .chains.hallucination_grader import hallucination_grader
from .chains.answer_grader import answer_grader
from .nodes import generate, grade_documents, retrieve, web_search
from .state import GraphState

load_dotenv(Path(__file__).parent / "../../.env")

def decide_to_generate(state: GraphState):
    print("--- ASSESS GRADED DOCUMENTS ---")

    if state["web_search"]:
        print("-- DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION")

        return WEB_SEARCH
    else:
        print("--- DECISION: GENERATE ---")

        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState):
    print("--- ASSESS GENERATION ---")
    question = state.get("question", "")
    documents = state.get("documents", "")
    generation = state.get("generation", "")

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})

    if score.binary_score == "yes":
        print("--- DECISION: GENERATION IS GROUNDED IN DOCUMENTS ---")
        print("--- GRADE GENERATION vs QUESTION --- ")

        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score == "yes":
            print("--- DECISION: GENERATION ADDRESS QUESTION ---")
            return "useful"
        else:
            print("--- DECISION: GENERATION DOES NOT ADDRESS QUESTION ---")
            return "not useful"
    else:
        print("--- DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS ---")
        return "not supported"

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, [WEB_SEARCH, GENERATE])
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, {
    "useful": END,
    "not useful": WEB_SEARCH,
    "not supported": GENERATE,
})
workflow.add_edge(WEB_SEARCH, GENERATE)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="flow.png")
