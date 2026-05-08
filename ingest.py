# Run this once to build the vector store before using the agent.
# Usage: python ingest.py
# Usage (custom PDF): python ingest.py path/to/your.pdf
#
# DESIGN NOTE (author's opinion): chunk_size=1000 / chunk_overlap=200 are sensible defaults
# but the right values depend on your documents and retrieval evals. Smaller chunks improve
# precision (less irrelevant text per chunk) but hurt recall (answer may span chunks).
# In production, run a retrieval eval (RAGAS or LangSmith evals) and tune from there.

import sys
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PDF = os.path.join(os.path.dirname(__file__), "docs", "attention_is_all_you_need.pdf")


def ingest(pdf_path: str = DEFAULT_PDF):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}\n"
            "Run: curl -L -o docs/attention_is_all_you_need.pdf "
            "https://arxiv.org/pdf/1706.03762"
        )

    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"  {len(docs)} pages loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        # RecursiveCharacterTextSplitter tries these separators in order:
        # paragraph → sentence → word → character.
        # This preserves semantic units better than a fixed-size split.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks after splitting")

    print("Building vector store (local embeddings, no API calls)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    print(f"Done. {len(chunks)} chunks indexed in ./chroma_db/")
    return vectorstore


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    ingest(pdf_path)
