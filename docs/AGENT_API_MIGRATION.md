# Agent API Migration Guide

The agent/session API has been updated to use vectorstore_name as a path parameter instead of a body parameter, consistent with all other memory APIs. This ensures proper memory isolation and follows the established API pattern.

## Breaking Changes

### 1. Session Creation

**OLD API:**
```javascript
POST /api/agent/session
{
  "system_prompt": "You are a helpful assistant",
  "vectorstore_name": "user_123_memories",  // Body parameter
  "config": {"use_memory": true}
}
```

**NEW API:**
```javascript
POST /api/agent/{vectorstore_name}/session
{
  "system_prompt": "You are a helpful assistant",
  "config": {"use_memory": true}
}
```

### 2. Sending Messages

**OLD API:**
```javascript
POST /api/agent/session/{session_id}
{
  "message": "Hello",
  "vectorstore_name": "user_123_memories"  // Body parameter (optional override)
}
```

**NEW API:**
```javascript
POST /api/agent/{vectorstore_name}/session/{session_id}
{
  "message": "Hello"
}
```

### 3. Session Management

**OLD APIs:**
```
GET /api/agent/session/{session_id}
DELETE /api/agent/session/{session_id}
GET /api/agent/sessions
```

**NEW APIs:**
```
GET /api/agent/{vectorstore_name}/session/{session_id}
DELETE /api/agent/{vectorstore_name}/session/{session_id}
GET /api/agent/{vectorstore_name}/sessions
```

## Migration Steps

### 1. Update Session Creation Code

```javascript
// OLD
const createSession = async (systemPrompt, vectorstore) => {
  return fetch('/api/agent/session', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      system_prompt: systemPrompt,
      vectorstore_name: vectorstore,
      config: {use_memory: true}
    })
  });
};

// NEW
const createSession = async (systemPrompt, vectorstore) => {
  return fetch(`/api/agent/${vectorstore}/session`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      system_prompt: systemPrompt,
      config: {use_memory: true}
    })
  });
};
```

### 2. Update Message Sending Code

```javascript
// OLD
const sendMessage = async (sessionId, message, vectorstore = null) => {
  const body = {message};
  if (vectorstore) body.vectorstore_name = vectorstore;
  
  return fetch(`/api/agent/session/${sessionId}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
};

// NEW
const sendMessage = async (sessionId, message, vectorstore) => {
  return fetch(`/api/agent/${vectorstore}/session/${sessionId}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message})
  });
};
```

### 3. Update Session Management Code

```javascript
// OLD
const getSession = async (sessionId) => {
  return fetch(`/api/agent/session/${sessionId}`);
};

const deleteSession = async (sessionId) => {
  return fetch(`/api/agent/session/${sessionId}`, {method: 'DELETE'});
};

const listSessions = async () => {
  return fetch('/api/agent/sessions');
};

// NEW
const getSession = async (sessionId, vectorstore) => {
  return fetch(`/api/agent/${vectorstore}/session/${sessionId}`);
};

const deleteSession = async (sessionId, vectorstore) => {
  return fetch(`/api/agent/${vectorstore}/session/${sessionId}`, {method: 'DELETE'});
};

const listSessions = async (vectorstore) => {
  return fetch(`/api/agent/${vectorstore}/sessions`);
};
```

## Key Benefits

1. **Consistency**: All memory APIs now use vectorstore_name as a path parameter
2. **Memory Isolation**: Sessions are strictly tied to their vectorstore
3. **Validation**: The API validates that session operations use the correct vectorstore
4. **Clarity**: The vectorstore is explicit in every request URL

## Error Handling

The new API will return a 400 error if you try to access a session with the wrong vectorstore:

```json
{
  "detail": "Session was created with vectorstore 'user_123' but request uses 'user_456'"
}
```

## Complete Example

```javascript
class AgentClient {
  constructor(baseUrl, vectorstore) {
    this.baseUrl = baseUrl;
    this.vectorstore = vectorstore;
  }

  async createSession(systemPrompt, config = {}) {
    const response = await fetch(`${this.baseUrl}/api/agent/${this.vectorstore}/session`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        system_prompt: systemPrompt,
        config: {use_memory: true, ...config}
      })
    });
    return response.json();
  }

  async sendMessage(sessionId, message, options = {}) {
    const response = await fetch(`${this.baseUrl}/api/agent/${this.vectorstore}/session/${sessionId}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message,
        ...options
      })
    });
    return response.json();
  }

  async getSession(sessionId) {
    const response = await fetch(`${this.baseUrl}/api/agent/${this.vectorstore}/session/${sessionId}`);
    return response.json();
  }

  async deleteSession(sessionId) {
    const response = await fetch(`${this.baseUrl}/api/agent/${this.vectorstore}/session/${sessionId}`, {
      method: 'DELETE'
    });
    return response.json();
  }

  async listSessions() {
    const response = await fetch(`${this.baseUrl}/api/agent/${this.vectorstore}/sessions`);
    return response.json();
  }
}

// Usage
const client = new AgentClient('http://localhost:5001', 'user_123_memories');
const session = await client.createSession('You are a helpful assistant');
const response = await client.sendMessage(session.session_id, 'Hello!');
```

This migration ensures proper memory isolation and follows the established API patterns used throughout the memory system.
