"""
Assembles the LangGraph StateGraph for the self-corrective RAG workflow.

Graph shape:

    START -> analyze_query -> retrieve -> grade_documents
                                              |
                        (conditional: decide_to_generate)
                        /          |            \\           \\
                  generate   rewrite_query   web_search    give_up
                     |             |             |             |
             check_hallucination   -> retrieve (loop)   -> generate    END
                  /        \\
        (conditional: decide_after_hallucination_check)
         end                 regenerate -> mark_regenerated -> generate
"""
from functools import lru_cache

from langgraph.graph import StateGraph, START, END

from graph.state import GraphState
from graph.nodes import (
    analyze_query,
    retrieve,
    grade_documents,
    decide_to_generate,
    rewrite_query,
    web_search,
    give_up,
    generate,
    check_hallucination,
    decide_after_hallucination_check,
    mark_regenerated,
)


@lru_cache(maxsize=1)
def build_workflow():
    graph = StateGraph(GraphState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("web_search", web_search)
    graph.add_node("give_up", give_up)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("mark_regenerated", mark_regenerated)

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
            "web_search": "web_search",
            "give_up": "give_up",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("give_up", END)

    graph.add_edge("generate", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        decide_after_hallucination_check,
        {
            "end": END,
            "regenerate": "mark_regenerated",
        },
    )
    graph.add_edge("mark_regenerated", "generate")

    return graph.compile()


def run_query(question: str, chat_history: list = None) -> dict:
    app = build_workflow()
    final_state = app.invoke({"question": question, "chat_history": chat_history or []})
    return {
        "question": question,
        "answer": final_state.get("generation"),
        "sources": final_state.get("sources", []),
        "query_type": final_state.get("query_type"),
        "retries_used": final_state.get("retry_count", 0),
        "used_web_search": final_state.get("used_web_search", False),
        "grounded": final_state.get("grounded"),
    }