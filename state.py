# DESIGN NOTE (author's opinion): web_search_needed as a string "yes"/"no" rather than bool
# is a pattern from LangGraph docs but a bool would be cleaner. Built your version here.
# If you refactor: change the field type and update decide_to_search() in nodes.py.

from typing import TypedDict, List
from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search_needed: str  # "yes" or "no"
