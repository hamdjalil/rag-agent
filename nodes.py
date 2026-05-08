# DESIGN NOTE (author's opinion): The grader uses a separate LLM call, which adds latency
# and cost per query. An alternative is a cross-encoder reranker (e.g. Cohere Rerank or
# a local sentence-transformers model) that's faster and doesn't burn LLM tokens.
# Built your grader-LLM version here — it's more readable for an interview walkthrough.
#
# DEPRECATION FIXED: Chroma moved from langchain_community.vectorstores to langchain_chroma
# in LangChain v0.2. Same API, just a different import path.
# DEPRECATION FIXED: TavilySearchResults moved from langchain_community to langchain_tavily
# in langchain-community 0.3.25. New class is TavilySearch.
#
# PROVIDER SWAP: Using Groq (free) instead of OpenAI for generation + grading.
# Using sentence-transformers (local, free) instead of OpenAI for embeddings.
# Interview note: "local embeddings keep inference costs at zero and remove a network
# dependency from the retrieval path — only generation and grading hit an external API."

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from state import AgentState


# Two models: grading needs reliable structured output (tool calling), generation just
# needs to follow instructions well. Larger model for grading, fast model for generation.
# Interview note: llama-3.1-8b-instant doesn't support tool calling reliably on Groq —
# with_structured_output() uses tool calling under the hood, so it needs a capable model.
grader_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
gen_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# --- Embeddings + Retriever (module-level singletons) ---
# Initialized once when nodes.py is first imported. Subsequent queries reuse the
# loaded model — no repeated weight loading.
# all-MiniLM-L6-v2: 384-dim, ~80MB, downloads once and caches to ~/.cache/huggingface/
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# --- Grader (structured output via Pydantic) ---
# with_structured_output() forces the LLM to return a typed object, not free text.
# This is more reliable than parsing "yes"/"no" out of a string.
class GradeDocuments(BaseModel):
    """Binary relevance score for retrieved documents."""
    binary_score: str = Field(description="'yes' if documents are relevant to the question, else 'no'")


grader_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You grade whether retrieved documents are relevant to a user question. "
        "Respond 'yes' if any document contains keywords or semantic meaning related to the question. "
        "Respond 'no' only if the documents are clearly off-topic. Be lenient — the goal is to "
        "filter irrelevant retrievals, not to demand perfect coverage.",
    ),
    ("human", "Documents:\n\n{documents}\n\nQuestion: {question}"),
])
grader = grader_prompt | grader_llm.with_structured_output(GradeDocuments)


# --- Generator ---
gen_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the user's question using only the provided context. "
        "Be concise and direct. If the context doesn't contain enough information "
        "to answer confidently, say so explicitly rather than guessing.",
    ),
    ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
])
generator = gen_prompt | gen_llm | StrOutputParser()


# --- Web search tool ---
web_search_tool = TavilySearch(max_results=3)


# ===== NODE FUNCTIONS =====
# Each node takes the full AgentState and returns a dict of keys to update.
# LangGraph merges the returned dict into state — keys not returned are unchanged.

def retrieve(state: AgentState) -> dict:
    """Pull k=4 documents from the vector store for the question."""
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


def grade_documents(state: AgentState) -> dict:
    """
    Grade retrieved docs for relevance. Sets web_search_needed='yes' if docs
    don't answer the question, so the conditional edge can route to web search.
    """
    docs_text = "\n\n".join(d.page_content for d in state["documents"])
    result = grader.invoke({
        "documents": docs_text,
        "question": state["question"],
    })
    needed = "no" if result.binary_score == "yes" else "yes"
    return {"web_search_needed": needed}


def web_search(state: AgentState) -> dict:
    """
    Tavily web search fallback. Appends web results to existing documents
    rather than replacing them — gives the generator both sources and lets it
    synthesize. In practice, the web results usually dominate since they're
    more relevant (that's why we're here).
    """
    raw = web_search_tool.invoke(state["question"])
    # TavilySearch.invoke() returns a dict: {"results": [...], "query": ..., ...}
    results = raw["results"] if isinstance(raw, dict) else raw
    web_docs = [
        Document(page_content=r["content"], metadata={"source": r.get("url", "")})
        for r in results
    ]
    return {"documents": state["documents"] + web_docs}


def generate(state: AgentState) -> dict:
    """Generate a grounded answer from whatever documents are in state."""
    context = "\n\n".join(d.page_content for d in state["documents"])
    answer = generator.invoke({
        "context": context,
        "question": state["question"],
    })
    return {"generation": answer}


# ===== ROUTING FUNCTION =====
# This is not a node — it's a pure function that reads state and returns
# the name of the next node. LangGraph calls it from add_conditional_edges().

def decide_to_search(state: AgentState) -> str:
    """Route to web_search if docs were irrelevant, else go straight to generate."""
    return "web_search" if state["web_search_needed"] == "yes" else "generate"
