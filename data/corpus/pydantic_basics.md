# Pydantic: Core Concepts

## What is Pydantic

Pydantic is a Python library for data validation and settings management using
Python type annotations. You describe the shape of your data as a class, and
Pydantic parses, validates, and coerces incoming data (such as JSON, dicts, or
environment variables) into instances of that class, raising clear validation
errors when the data does not match.

## Defining Models

A Pydantic model is a class that inherits from `BaseModel`. Each class attribute,
annotated with a type, becomes a model field.

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    signup_ts: str | None = None
    friends: list[int] = []
```

When you instantiate `User(id="123", name="Alice", friends=["1", "2"])`,
Pydantic will coerce the string `"123"` to an `int`, and each string in
`friends` to an `int`, because that is what the type annotations declare.

## Validation Errors

If the input data cannot be coerced into the declared types, Pydantic raises a
`ValidationError`. This exception contains a list of every field that failed,
the location of the failure, and a human-readable message, which makes it easy
to return meaningful error messages to API clients.

```python
from pydantic import ValidationError

try:
    User(id="not-a-number", name="Bob")
except ValidationError as e:
    print(e.errors())
```

## Field Customization

The `Field` function lets you add extra constraints and metadata to a field,
such as default values, minimum/maximum values, regex patterns, or
descriptions used in generated documentation.

```python
from pydantic import Field

class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0, description="Price in USD")
```

## Custom Validators

For validation logic that goes beyond simple type constraints, Pydantic supports
custom validators through the `field_validator` decorator (in Pydantic v2) or
`validator` (in Pydantic v1). These let you write arbitrary Python code to
check or transform a field's value.

```python
from pydantic import field_validator

class Account(BaseModel):
    username: str

    @field_validator("username")
    @classmethod
    def username_must_be_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError("username must be alphanumeric")
        return v
```

## Nested Models and Serialization

Pydantic models can be nested inside one another, and Pydantic will validate
and construct the nested structures recursively. Models can be converted back
into plain Python dictionaries with `.model_dump()`, or into a JSON string with
`.model_dump_json()`, which is what frameworks like FastAPI use to serialize
responses.

## Settings Management

Pydantic also provides a `BaseSettings` class (in the separate
`pydantic-settings` package for v2) that reads configuration values from
environment variables or `.env` files, using the same declarative style as a
normal model. This is a common pattern for centralizing application
configuration such as API keys, database URLs, and feature flags.
