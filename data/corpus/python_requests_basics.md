# Python Requests Library: Core Concepts

## What is Requests

Requests is a widely used Python library that provides a simple, human-friendly
interface for making HTTP requests. It wraps the lower-level `urllib3` library
and hides most of the boilerplate needed to send requests and read responses.

## Making a Basic Request

The library exposes one function per HTTP method: `requests.get`,
`requests.post`, `requests.put`, `requests.delete`, and so on. Each returns a
`Response` object.

```python
import requests

response = requests.get("https://api.example.com/items")
print(response.status_code)
print(response.json())
```

The `Response` object exposes `.status_code` (the HTTP status code as an
integer), `.text` (the raw response body as a string), `.json()` (the body
parsed as JSON, raising an error if it is not valid JSON), and `.headers` (a
case-insensitive dictionary of response headers).

## Query Parameters and Request Bodies

Query parameters can be passed as a dictionary via the `params` argument
rather than manually building a query string. For request bodies, `json=`
automatically serializes a Python dictionary to JSON and sets the
`Content-Type` header, while `data=` sends form-encoded or raw data instead.

```python
requests.get("https://api.example.com/search", params={"q": "fastapi"})
requests.post("https://api.example.com/items", json={"name": "widget"})
```

## Headers and Authentication

Custom headers, such as an `Authorization` bearer token, are passed as a
dictionary via the `headers` argument. Requests also has built-in helpers for
common authentication schemes, such as `requests.auth.HTTPBasicAuth`, and
supports a shorthand `auth=(username, password)` tuple for basic auth.

```python
headers = {"Authorization": "Bearer my-token"}
requests.get("https://api.example.com/me", headers=headers)
```

## Error Handling

By default, Requests does not raise an exception for HTTP error status codes
like 404 or 500; it only raises exceptions for connection-level problems (such
as a DNS failure or a timeout). To treat error status codes as exceptions, you
call `response.raise_for_status()`, which raises an `HTTPError` if the status
code indicates a client or server error.

```python
response = requests.get("https://api.example.com/items/999")
response.raise_for_status()  # raises HTTPError if status is 4xx or 5xx
```

## Sessions

A `requests.Session` object persists certain parameters (such as cookies and
headers) across multiple requests to the same host, and reuses the underlying
TCP connection, which improves performance when making several requests to
the same server.

```python
session = requests.Session()
session.headers.update({"Authorization": "Bearer my-token"})
session.get("https://api.example.com/items")
session.get("https://api.example.com/orders")
```

## Timeouts

It is strongly recommended to always pass a `timeout` argument to requests
calls; without one, a request can hang indefinitely if the server never
responds, since Requests has no default timeout.

```python
requests.get("https://api.example.com/items", timeout=5)
```
