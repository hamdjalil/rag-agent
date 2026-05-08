import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph import build_graph

st.set_page_config(
    page_title="RAG Agent",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Sidebar tutorial (auto-opens on first visit; user can collapse anytime) ---
with st.sidebar:
    st.header("Tutorial")
    st.markdown("Welcome! Here's what this agent does and how to use it.")

    st.subheader("How it works")
    st.markdown("""
1. **Retrieve** — searches a local [Chroma](https://www.trychroma.com/) vector store
   for the 4 most semantically similar chunks to your question.

2. **Grade** — an LLM judges whether the retrieved chunks are actually relevant.
   If not, it routes to web search to avoid a hallucinated answer.

3. **Web search (fallback)** — if local docs weren't relevant,
   [Tavily](https://tavily.com/) fetches live web results and appends them to the context.

4. **Generate** — a second LLM produces a grounded answer from whichever
   documents ended up in context.
""")

    st.divider()

    st.subheader("What's in the vector store")
    st.markdown("""
The store is indexed from **"Attention Is All You Need"** (Vaswani et al., 2017) —
the original Transformer paper.

Try asking:
""")
    example_questions = [
        "What is multi-head attention?",
        "How does positional encoding work?",
        "What are the encoder and decoder stacks made of?",
        "Why did the authors replace recurrence with attention?",
    ]
    for q in example_questions:
        st.markdown(f"- *{q}*")

    st.info(
        "For questions outside the paper's scope, the agent automatically "
        "falls back to a live Tavily web search.",
        icon="ℹ️",
    )

    st.divider()

    st.subheader("Add your own documents")
    st.markdown("""
Run this once to index a custom PDF:

```bash
python ingest.py path/to/your.pdf
```
""")

# --- Main app ---
st.title("RAG Agent")
st.caption("Retrieves from a local vector store · falls back to Tavily web search if docs aren't relevant.")

if "graph" not in st.session_state:
    with st.spinner("Compiling graph..."):
        st.session_state.graph = build_graph()

SUGGESTIONS = [
    "What is multi-head attention?",
    "How does positional encoding work?",
    "How does attention differ from CNNs?",
]

# Apply a pending suggestion from a bubble click before the widget renders
if st.session_state.get("_pending_question"):
    st.session_state.question_input = st.session_state.pop("_pending_question")

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

st.markdown("""
<style>
/* Column gap so bubbles don't bleed into each other */
div[data-testid="stHorizontalBlock"] {
    gap: 10px !important;
}
/* Suggestion bubble buttons */
div[data-testid="stHorizontalBlock"] [data-testid^="stBaseButton"],
div[data-testid="stHorizontalBlock"] .stButton > button {
    border-radius: 999px !important;
    border: 1px solid rgba(250, 75, 75, 0.45) !important;
    background: transparent !important;
    font-size: 0.8rem !important;
    padding: 6px 16px !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
    transition: background 0.15s, border-color 0.15s !important;
}
div[data-testid="stHorizontalBlock"] [data-testid^="stBaseButton"]:hover,
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: rgba(250, 75, 75, 0.08) !important;
    border-color: #ff4b4b !important;
}
div[data-testid="stHorizontalBlock"] [data-testid^="stBaseButton"]:active,
div[data-testid="stHorizontalBlock"] .stButton > button:active {
    background: rgba(250, 75, 75, 0.18) !important;
}
</style>
""", unsafe_allow_html=True)

question = st.text_input(
    label="Ask a question",
    placeholder="What is multi-head attention?",
    key="question_input",
)

# Suggestion bubbles — clicking populates the input and immediately runs the graph
cols = st.columns(len(SUGGESTIONS))
for col, suggestion in zip(cols, SUGGESTIONS):
    with col:
        if st.button(suggestion, use_container_width=True):
            st.session_state._pending_question = suggestion
            st.rerun()

question = st.session_state.question_input

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
