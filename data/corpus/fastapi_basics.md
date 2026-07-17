# FastAPI: Core Concepts

## What is FastAPI

FastAPI is a modern Python web framework for building APIs. It is built on top of
Starlette (for the web parts) and Pydantic (for the data parts). The main selling
points are speed of development, automatic interactive documentation, and strong
type-checking support using standard Python type hints.

## Path Operations

In FastAPI, an endpoint is called a "path operation". You declare one by decorating
a function with an HTTP method decorator such as `@app.get`, `@app.post`,
`@app.put`, `@app.delete`, or `@app.patch`, and passing the URL path as an
argument.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

Path parameters (like `item_id` above) are declared directly in the path string
using curly braces, and FastAPI converts and validates them automatically based
on the type hint. Query parameters are simply extra function arguments that are
not part of the path.

## Request Bodies with Pydantic

To accept a JSON request body, define a Pydantic model and use it as a type hint
for a parameter in your path operation function. FastAPI will read the body,
validate it against the model, convert types, and give you a normal Python object
to work with. If validation fails, FastAPI automatically returns a 422 response
with details about what went wrong.

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

## Dependency Injection

FastAPI has a dependency injection system built around the `Depends` function.
A dependency is just a callable (often a function) that FastAPI will call for
you and inject the result into your path operation. This is commonly used for
shared logic such as database sessions, authentication checks, or pagination
parameters, so that this logic does not need to be repeated in every endpoint.

```python
from fastapi import Depends

def common_params(skip: int = 0, limit: int = 100):
    return {"skip": skip, "limit": limit}

@app.get("/users/")
async def list_users(params: dict = Depends(common_params)):
    return params
```

## Automatic Documentation

Because FastAPI relies on type hints and Pydantic models, it can generate an
OpenAPI schema automatically. Two interactive documentation UIs are provided out
of the box: Swagger UI at `/docs` and ReDoc at `/redoc`. No extra configuration
is required to get these; they are derived directly from your route definitions,
type hints, and docstrings.

## Async Support

Path operation functions can be declared with `async def` or a plain `def`.
FastAPI handles both correctly. Using `async def` is recommended when the
function performs I/O-bound work (such as calling an external API or database)
using an async-compatible library, since this allows the server to handle other
requests while waiting. If you only have synchronous libraries available,
regular `def` functions are still fully supported and FastAPI runs them in a
thread pool so they do not block the event loop.

## Error Handling

FastAPI provides an `HTTPException` class for returning custom error responses
with a specific status code and detail message. You can also register custom
exception handlers with `@app.exception_handler(SomeException)` to control how
specific exception types are converted into HTTP responses across your whole
application.

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in db:
        raise HTTPException(status_code=404, detail="Item not found")
    return db[item_id]
```
