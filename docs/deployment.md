# Deploying Agents as REST APIs

Kawai allows you to deploy your agents as REST API servers, making them accessible from other applications or integrable into existing infrastructure.

## Quick Start

Deploy an agent with a single method call:

```python
import weave
from kawai import KawaiReactAgent, WebSearchTool, OpenAIModel

weave.init(project_name="kawai-server")

model = OpenAIModel(
    model_id="google/gemini-3-flash-preview",
    base_url="https://openrouter.ai/api/v1",
    api_key_env_var="OPENROUTER_API_KEY",
)

agent = KawaiReactAgent(
    model=model,
    tools=[WebSearchTool()],
    max_steps=10,
)

# Start the REST API server
agent.serve(port=8000)
```

The server provides these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Non-streaming chat - send a message, get the response |
| `/stream` | GET | Server-Sent Events for real-time streaming |
| `/health` | GET | Health check with status and session count |
| `/sessions/{id}` | GET | Get session information |
| `/sessions/{id}` | DELETE | Delete a session |

## API Reference

### POST /chat

Send a message to the agent and receive the complete response.

**Request Body:**
```json
{
    "message": "What is the capital of France?",
    "session_id": "optional-session-id",
    "max_steps": 5,
    "force_provide_answer": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | The message/prompt for the agent |
| `session_id` | string | No | Session ID for conversation continuity |
| `max_steps` | integer | No | Override default max_steps |
| `force_provide_answer` | boolean | No | Force answer if max_steps exhausted (default: true) |

**Response:**
```json
{
    "answer": "The capital of France is Paris.",
    "session_id": "abc123",
    "steps": 2,
    "completed": true,
    "plan": null
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is machine learning?"}'
```

**Example with Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "What is machine learning?"}
)
result = response.json()
print(result["answer"])
```

### GET /stream

Connect to receive real-time events during agent execution via Server-Sent Events (SSE).

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | Yes | The message/prompt for the agent |
| `session_id` | string | No | Session ID for conversation continuity |
| `max_steps` | integer | No | Override default max_steps |
| `force_provide_answer` | boolean | No | Force answer if max_steps exhausted (default: true) |

**Event Types:**

| Type | Description | Data Fields |
|------|-------------|-------------|
| `run_start` | Agent started processing | `prompt`, `model` |
| `step_start` | New step began | `step_index` |
| `reasoning` | Agent's thinking | `reasoning` |
| `tool_call` | Tool being called | `tool_name`, `tool_arguments` |
| `tool_result` | Tool returned result | `tool_name`, `tool_result` |
| `planning` | Plan generated/updated | `plan`, `updated_plan` |
| `warning` | Warning occurred | `message` |
| `run_end` | Agent finished | `answer` |

**Example with curl:**
```bash
curl -N "http://localhost:8000/stream?message=Search%20for%20AI%20news"
```

**Example with Python:**
```python
import requests
import json

response = requests.get(
    "http://localhost:8000/stream",
    params={"message": "Search for AI news"},
    stream=True
)

for line in response.iter_lines():
    if line:
        data = line.decode().removeprefix("data: ")
        event = json.loads(data)
        print(f"{event['type']}: {event['data']}")
```

**Example with JavaScript (browser):**
```javascript
const eventSource = new EventSource(
    "http://localhost:8000/stream?message=" + encodeURIComponent("Hello!")
);

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.type, data.data);
    
    if (data.type === "run_end") {
        eventSource.close();
    }
};
```

### GET /health

Check server health status.

**Response:**
```json
{
    "status": "healthy",
    "sessions": 5,
    "uptime": 3600.5,
    "version": "0.1.0"
}
```

### GET /sessions/{session_id}

Get information about a specific session.

**Response:**
```json
{
    "session_id": "abc123",
    "created_at": "2024-01-15T10:30:00",
    "last_accessed": "2024-01-15T10:35:00",
    "message_count": 6
}
```

### DELETE /sessions/{session_id}

Delete a session to clear its conversation history.

**Response:**
```json
{
    "session_id": "abc123",
    "deleted": true
}
```

## Session Management

Sessions maintain conversation memory across requests, enabling multi-turn conversations:

```python
import requests

SESSION_ID = "user-123"

# First message
r1 = requests.post(
    "http://localhost:8000/chat",
    json={"message": "What is Python?", "session_id": SESSION_ID}
)
print(r1.json()["answer"])

# Follow-up (agent remembers context)
r2 = requests.post(
    "http://localhost:8000/chat",
    json={"message": "What are its main features?", "session_id": SESSION_ID}
)
print(r2.json()["answer"])
```

Sessions automatically expire after the configured timeout (default: 1 hour).

## Configuration Options

The `serve()` method accepts these parameters:

```python
agent.serve(
    host="0.0.0.0",          # Bind to all interfaces
    port=8000,               # Server port
    session_timeout=3600,    # Session expiry in seconds (1 hour)
    enable_cors=True,        # Enable CORS for browsers
    allowed_origins=["*"],   # CORS allowed origins
    log_level="info",        # Uvicorn log level
)
```

## Production Deployment

### Using a Process Manager

For production, use a process manager like systemd or supervisord:

```ini
# /etc/systemd/system/kawai-agent.service
[Unit]
Description=Kawai Agent API
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/kawai
Environment="OPENROUTER_API_KEY=your-key"
Environment="SERPER_API_KEY=your-key"
ExecStart=/opt/kawai/.venv/bin/python -m examples.serve_example
Restart=always

[Install]
WantedBy=multi-user.target
```

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY kawai ./kawai
COPY examples ./examples

# Install dependencies
RUN uv sync

# Environment variables
ENV OPENROUTER_API_KEY=""
ENV SERPER_API_KEY=""

EXPOSE 8000

CMD ["uv", "run", "examples/serve_example.py"]
```

```bash
docker build -t kawai-agent .
docker run -p 8000:8000 \
    -e OPENROUTER_API_KEY=your-key \
    -e SERPER_API_KEY=your-key \
    kawai-agent
```

### Reverse Proxy with nginx

For production, put the server behind nginx:

```nginx
upstream kawai {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://kawai;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # For SSE streaming
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
    }
}
```

## Security Considerations

1. **API Keys**: Never expose API keys in client-side code. Use environment variables.

2. **Rate Limiting**: Consider adding rate limiting middleware for production:
   ```python
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   ```

3. **Authentication**: Add authentication for production deployments:
   ```python
   from fastapi import Depends, HTTPException
   from fastapi.security import APIKeyHeader
   
   api_key_header = APIKeyHeader(name="X-API-Key")
   
   async def verify_api_key(api_key: str = Depends(api_key_header)):
       if api_key != os.getenv("API_KEY"):
           raise HTTPException(status_code=403)
   ```

4. **HTTPS**: Always use HTTPS in production. Configure SSL/TLS in your reverse proxy.

5. **CORS**: Restrict `allowed_origins` to your specific domains in production.
