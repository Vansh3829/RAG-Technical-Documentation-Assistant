# LangGraph: Core Concepts

## What is LangGraph

LangGraph is a library for building stateful, multi-step applications with
language models, expressed as a graph of nodes and edges. Instead of writing a
single long chain of calls, you describe your application as a `StateGraph`:
a set of nodes (each a Python function that receives and returns a piece of
shared state) connected by edges that define the order of execution, including
conditional branches and loops.

## State Schema

Every LangGraph application defines a state schema, usually a `TypedDict` or a
Pydantic model, that describes what data is passed between nodes.

```python
from typing import TypedDict

class GraphState(TypedDict):
    question: str
    documents: list
    generation: str
```

Each node function receives the current state and returns a dictionary of the
fields it wants to update. LangGraph merges this partial update into the
overall state before passing it to the next node. This makes it straightforward
to trace exactly what data is available at each step of a workflow.

## Nodes and Edges

A node is added to the graph with `graph.add_node("name", function)`. Edges
connect nodes and are added with `graph.add_edge("from_node", "to_node")` for
a direct, unconditional transition. The graph also needs a designated entry
point, set with `graph.set_entry_point("first_node")`, and normally ends by
routing to the special `END` marker.

## Conditional Edges

Conditional edges allow the graph to branch based on the current state. You
provide a routing function that inspects the state and returns the name of the
next node (or a symbolic key that is mapped to a node name).

```python
def decide_next_step(state):
    if state["documents"]:
        return "generate"
    return "retry"

graph.add_conditional_edges(
    "grade_documents",
    decide_next_step,
    {"generate": "generate_node", "retry": "rewrite_node"},
)
```

This pattern is what enables self-correcting workflows: a node can grade or
check the quality of intermediate results, and the conditional edge decides
whether to proceed forward or loop back and try again.

## Cycles and Retry Limits

Because LangGraph graphs can contain cycles (a conditional edge can route back
to an earlier node), it is important to track loop counters inside the state,
such as a `retry_count` field, and check that counter in the routing function
to avoid infinite loops. A typical pattern increments a counter each time a
retry path is taken, and the routing function falls back to a different branch
(such as a "give up" or "use a fallback" node) once a maximum number of
retries has been reached.

## Compiling and Running

Once all nodes and edges are added, calling `graph.compile()` returns a
runnable application. This can be invoked synchronously with `.invoke(state)`,
which runs the graph to completion and returns the final state, or streamed
with `.stream(state)`, which yields intermediate state updates as each node
finishes, which is useful for showing progress in a user interface.

## Common Use Cases

LangGraph is frequently used to implement patterns such as: self-corrective
retrieval-augmented generation (retrieve, grade relevance, and re-retrieve if
needed), multi-agent systems (where different nodes represent different
specialized agents that hand off control to one another), and tool-using
agents that loop between "decide which tool to call" and "execute the tool"
until a final answer is produced.
