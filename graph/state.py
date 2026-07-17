"""
The shared state schema that flows through every node in the LangGraph
workflow.

Design notes (see README "Design Decisions" for the full write-up):
- `retry_count` / `max_retries` implement the retry limit for the
  rewrite-and-re-retrieve loop, so the graph can never spin forever.
- `original_question` is kept separate from `question` so the final answer
  can always reference what the user actually asked, even after the query
  has been rewritten one or more times internally.
- `documents` holds every chunk retrieved this pass; `graded_documents` holds
  only the ones the grading node judged relevant. Keeping both makes the
  workflow easy to debug/trace (you can see what was retrieved vs. what was
  actually used).
- `used_web_search` / `regenerated` are simple booleans that prevent the
  hallucination-check loop from also running forever (max one regeneration).
"""
from typing import List, Optional, TypedDict


class GradedDocument(TypedDict):
    content: str
    source: str
    chunk_index: Optional[int]
    score: float
    relevant: bool


class GraphState(TypedDict, total=False):
    # the question, as it evolves
    original_question: str
    question: str
    query_type: Optional[str]

    # bonus: conversation memory -- past (question, answer) pairs from this
    # session, oldest first. Used by analyze_query to resolve references like
    # "it" / "that library" from earlier in the conversation. Not used by
    # generate() directly, to keep the generation prompt/context small.
    chat_history: List[tuple]

    # retrieval
    documents: List[GradedDocument]
    graded_documents: List[GradedDocument]

    # self-correction bookkeeping
    retry_count: int
    max_retries: int
    used_web_search: bool
    regenerated: bool

    # output
    generation: Optional[str]
    sources: List[dict]
    grounded: Optional[bool]
    route: Optional[str]  # last routing decision, kept for observability