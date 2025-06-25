# Minsky Memory Agent Examples

## Travel Agent Scenario

This example demonstrates how the three-layer API works together to create an intelligent travel agent that learns and remembers user preferences.

### Step 1: Store Fundamental Memories (NEME API)

First, we store atomic memories about the user:

```javascript
// Store basic preferences as Nemes
await fetch('/api/memory', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: "User prefers window seats on flights",
    apply_grounding: true
  })
});

await fetch('/api/memory', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: "User enjoys Michelin star restaurants",
    apply_grounding: true
  })
});

await fetch('/api/memory', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: "User's wife is vegetarian",
    apply_grounding: true
  })
});
```

### Step 2: Construct Mental States (K-LINE API)

Now we can construct mental states for specific queries:

```javascript
// Construct mental state for flight booking
const flightMemories = await fetch('/api/klines/recall', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "flight booking preferences",
    top_k: 5
  })
});

// Construct mental state with LLM filtering for higher quality results
const filteredMemories = await fetch('/api/klines/recall', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "restaurant preferences for dinner",
    top_k: 10,
    memory_type: "neme",  // Only search atomic memories (nemes)
    use_llm_filtering: true  // Apply intelligent relevance filtering
  })
});

// Search for previous reasoning patterns (k-lines) about similar topics
const previousReasoning = await fetch('/api/klines/recall', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "restaurant recommendations",
    top_k: 5,
    memory_type: "k-line"  // Only search previous reasoning (k-lines)
    // This uses Redis FILTER: @type == 'k-line' for efficient server-side filtering
  })
});

// You can also combine memory_type with custom filters
const complexSearch = await fetch('/api/memory/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "travel preferences",
    top_k: 10,
    memory_type: "neme",  // Only nemes
    filter: "@tags == 'preferences'"  // Additional custom filter
    // Combined Redis FILTER: (@tags == "preferences") && ((!exists(@type) || @type == 'neme'))
  })
});

// Answer specific questions using K-line reasoning
const seatRecommendation = await fetch('/api/klines/answer', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "What seat should I book for my upcoming flight?",
    top_k: 5
  })
});

console.log(await seatRecommendation.json());
// Output: {
//   "answer": "I recommend booking a window seat for your flight.",
//   "confidence": "I'm fairly confident",
//   "reasoning": "Based on your stored preferences, you prefer window seats on flights.",
//   "supporting_memories": [...]
// }

// Handle mixed memory types in search results
const allMemories = await fetch('/api/memory/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "travel preferences",
    top_k: 10
    // No memory_type specified - returns both nemes and k-lines
  })
});

const result = await allMemories.json();
console.log(`Found ${result.memory_breakdown.nemes} nemes and ${result.memory_breakdown.klines} k-lines`);

// Process different memory types appropriately
result.memories.forEach(memory => {
  if (memory.type === 'k-line') {
    console.log(`K-line: ${memory.original_question} → ${memory.answer}`);
  } else {
    console.log(`Neme: ${memory.text}`);
  }
});

## Performance Benefits of Redis Filtering

The memory type filtering is implemented using Redis VSIM FILTER commands for optimal performance:

- **Server-side filtering**: Filtering happens at the Redis level in C code, not in Python
- **Reduced network traffic**: Only matching results are returned over the network
- **Scalable**: Performance remains consistent even with large memory collections
- **Efficient indexing**: Leverages Redis's optimized metadata indexing
- **Combined filters**: Multiple filter conditions are combined efficiently

### Filter Examples:
- `memory_type: "neme"` → `(!exists(@type) || @type == 'neme')`
- `memory_type: "k-line"` → `@type == 'k-line'`
- Combined: `(@tags == "travel") && (@type == 'k-line')`
```

### Step 3: Full Agent Conversation (AGENT API)

Finally, we can have natural conversations that automatically use memory:

```javascript
// Create an agent session
const session = await fetch('/api/agent/session', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    system_prompt: "You are a helpful travel assistant with access to user preferences and history.",
    config: {
      use_memory: true,
      model: "gpt-3.5-turbo",
      temperature: 0.7
    }
  })
});

const { session_id } = await session.json();

// Have a conversation that automatically uses stored memories
const response = await fetch(`/api/agent/session/${session_id}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "I'm planning a romantic dinner for my wife and me. Any suggestions?"
  })
});

const result = await response.json();
console.log(result.message);
// Output: "I'd recommend looking for vegetarian-friendly Michelin star restaurants, 
//          since I know your wife is vegetarian and you enjoy fine dining experiences..."
```

## Memory Extraction Example

The K-LINE API can also extract new memories from conversations:

```javascript
// Extract memories from a conversation
const extraction = await fetch('/api/klines/extract', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    raw_input: `User: I had an amazing experience at Le Bernardin last week. 
                The seafood was incredible, though my wife couldn't eat most of it 
                since she's vegetarian. We sat by the window and had a great view.
                I think I prefer restaurants with good views now.`,
    context_prompt: "Extract dining preferences, experiences, and family information",
    apply_grounding: true
  })
});

const extractionResult = await extraction.json();
console.log(extractionResult);
// This will automatically create new Nemes like:
// - "User enjoyed Le Bernardin restaurant"
// - "User prefers restaurants with good views"
// - "Wife is vegetarian (confirmed)"
```

## Progressive Development Pattern

### 1. Start Simple (Nemes)
```javascript
// Begin with basic memory storage
const memories = [
  "User budget is $500 per night for hotels",
  "User travels frequently for business",
  "User has TSA PreCheck"
];

for (const memory of memories) {
  await storeNeme(memory);
}
```

### 2. Add Reasoning (K-lines)
```javascript
// Build mental states for specific tasks
const hotelPreferences = await constructMentalState("hotel booking");
const answer = await answerWithReasoning("What's my hotel budget?");
```

### 3. Full Intelligence (Agent)
```javascript
// Deploy complete conversational agent
const agent = await createAgentSession({
  system_prompt: "Travel concierge with memory of user preferences",
  use_memory: true
});

// Agent automatically uses Nemes and K-lines internally
await chatWithAgent(agent, "Plan my business trip to Tokyo");
```

## Error Handling

```javascript
try {
  const response = await fetch('/api/memory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: "User prefers aisle seats",
      apply_grounding: true
    })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  const result = await response.json();
  
  if (!result.success) {
    console.error('Memory storage failed:', result.error);
  } else {
    console.log('Neme stored:', result.memory_id);
  }
} catch (error) {
  console.error('Network error:', error);
}
```

## Best Practices

1. **Start with Nemes**: Build your knowledge base with atomic memories
2. **Use K-lines for reasoning**: Construct mental states for complex queries
3. **Deploy Agents for UX**: Provide natural conversation interfaces
4. **Extract continuously**: Use conversations to grow your memory base
5. **Apply grounding**: Let the system add temporal and spatial context
6. **Filter appropriately**: Use Redis filters for domain-specific queries

This layered approach allows you to build sophisticated memory-enabled applications while maintaining clear separation of concerns and theoretical grounding.
