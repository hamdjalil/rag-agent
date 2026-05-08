# Test each node in isolation before wiring the full graph.
# This lets you debug retrieval quality, grader accuracy, and generation
# independently — much easier than debugging inside LangGraph's executor.
#
# Usage: python test_nodes.py
# Requires: chroma_db/ to exist (run ingest.py first), and .env with keys.

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from nodes import retrieve, grade_documents, web_search, generate

# A question well-covered by the paper
IN_CORPUS_Q = "What is the attention mechanism in transformers?"

# A question the paper won't answer
OUT_OF_CORPUS_Q = "What is the current price of Bitcoin?"

SEP = "-" * 60


def test_retrieve():
    print(f"\n{'='*60}")
    print("NODE: retrieve")
    print(f"Question: {IN_CORPUS_Q}")
    print(SEP)

    state = {"question": IN_CORPUS_Q, "documents": [], "generation": "", "web_search_needed": ""}
    result = retrieve(state)
    docs = result["documents"]

    print(f"Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"\n[Doc {i}] source={source}")
        print(doc.page_content[:300].strip() + "...")
    return docs


def test_grade_relevant(docs):
    print(f"\n{'='*60}")
    print("NODE: grade_documents (expecting: relevant)")
    print(f"Question: {IN_CORPUS_Q}")
    print(SEP)

    state = {
        "question": IN_CORPUS_Q,
        "documents": docs,
        "generation": "",
        "web_search_needed": "",
    }
    result = grade_documents(state)
    print(f"web_search_needed = '{result['web_search_needed']}'  (expected: 'no')")
    return result["web_search_needed"]


def test_grade_irrelevant():
    print(f"\n{'='*60}")
    print("NODE: grade_documents (expecting: irrelevant)")
    print(f"Question: {OUT_OF_CORPUS_Q}")
    print(SEP)

    # Simulate retrieving documents that are clearly off-topic
    dummy_docs = [
        Document(page_content="The transformer architecture uses self-attention to process sequences."),
        Document(page_content="Multi-head attention allows the model to attend to different positions."),
    ]
    state = {
        "question": OUT_OF_CORPUS_Q,
        "documents": dummy_docs,
        "generation": "",
        "web_search_needed": "",
    }
    result = grade_documents(state)
    print(f"web_search_needed = '{result['web_search_needed']}'  (expected: 'yes')")
    return result["web_search_needed"]


def test_web_search():
    print(f"\n{'='*60}")
    print("NODE: web_search")
    print(f"Question: {OUT_OF_CORPUS_Q}")
    print(SEP)

    state = {
        "question": OUT_OF_CORPUS_Q,
        "documents": [],
        "generation": "",
        "web_search_needed": "yes",
    }
    result = web_search(state)
    docs = result["documents"]
    print(f"Web search returned {len(docs)} results")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"\n[Result {i}] source={source}")
        print(doc.page_content[:300].strip() + "...")
    return docs


def test_generate_from_corpus(docs):
    print(f"\n{'='*60}")
    print("NODE: generate (from corpus docs)")
    print(f"Question: {IN_CORPUS_Q}")
    print(SEP)

    state = {
        "question": IN_CORPUS_Q,
        "documents": docs,
        "generation": "",
        "web_search_needed": "no",
    }
    result = generate(state)
    print(f"Answer:\n{result['generation']}")


def test_generate_from_web(web_docs):
    print(f"\n{'='*60}")
    print("NODE: generate (from web search docs)")
    print(f"Question: {OUT_OF_CORPUS_Q}")
    print(SEP)

    state = {
        "question": OUT_OF_CORPUS_Q,
        "documents": web_docs,
        "generation": "",
        "web_search_needed": "yes",
    }
    result = generate(state)
    print(f"Answer:\n{result['generation']}")


if __name__ == "__main__":
    print("Testing nodes in isolation...")
    print("(Each node is called directly — no LangGraph execution yet)\n")

    # Retrieve
    corpus_docs = test_retrieve()

    # Grade: relevant path
    test_grade_relevant(corpus_docs)

    # Grade: irrelevant path
    test_grade_irrelevant()

    # Web search
    web_docs = test_web_search()

    # Generate: from corpus
    test_generate_from_corpus(corpus_docs)

    # Generate: from web
    test_generate_from_web(web_docs)

    print(f"\n{'='*60}")
    print("All node tests complete.")
