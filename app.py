import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph import build_graph

st.set_page_config(page_title="RAG Agent", page_icon="🔍", layout="centered")
st.title("RAG Agent with Web Search Fallback")
st.caption("Retrieves from a local vector store, falls back to Tavily web search if docs aren't relevant.")

if "graph" not in st.session_state:
    with st.spinner("Compiling graph..."):
        st.session_state.graph = build_graph()

question = st.text_input("Ask a question:", placeholder="What is multi-head attention?")

if question:
    with st.spinner("Running graph..."):
        steps = []
        for step in st.session_state.graph.stream({"question": question}):
            for node_name, node_output in step.items():
                steps.append((node_name, node_output))

    # Display execution trace
    st.subheader("Execution trace")
    for node_name, node_output in steps:
        with st.expander(f"Node: `{node_name}`", expanded=(node_name == "generate")):
            if node_name == "retrieve":
                docs = node_output.get("documents", [])
                st.write(f"Retrieved **{len(docs)}** documents from vector store.")
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("source", "unknown")
                    st.markdown(f"**Doc {i}** — `{src}`")
                    st.text(doc.page_content[:400])

            elif node_name == "grade":
                needed = node_output.get("web_search_needed", "?")
                if needed == "yes":
                    st.warning("Docs not relevant — routing to **web search**.")
                else:
                    st.success("Docs are relevant — routing to **generate**.")

            elif node_name == "web_search":
                docs = node_output.get("documents", [])
                st.write(f"Web search returned **{len(docs)}** results (appended to state).")
                for doc in docs:
                    src = doc.metadata.get("source", "")
                    st.markdown(f"- {src}")

            elif node_name == "generate":
                answer = node_output.get("generation", "")
                st.markdown(answer)

    # Final answer prominent display
    final = next(
        (out.get("generation") for _, out in reversed(steps) if "generation" in out),
        None,
    )
    if final:
        st.divider()
        st.subheader("Answer")
        st.markdown(final)
