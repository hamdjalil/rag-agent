from langgraph.graph import StateGraph, START, END

from state import AgentState
from nodes import retrieve, grade_documents, web_search, generate, decide_to_search


def build_graph():
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    # Fixed edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")

    # Conditional edge: grade → web_search or grade → generate
    workflow.add_conditional_edges(
        "grade",
        decide_to_search,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()
    print(graph.get_graph().draw_ascii())
