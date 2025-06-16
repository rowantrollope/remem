# Contextual Grounding System

The Memory Agent now includes an advanced **Contextual Grounding System** that automatically resolves context-dependent references in memories, making them more useful and searchable over time.

## The Problem

When storing memories, people often use context-dependent references that become unclear later:

- **Temporal**: "It's my birthday today" → unclear what "today" means later
- **Spatial**: "It's really hot outside" → unclear where "outside" refers to
- **Personal**: "This guy is annoying" → unclear who "this guy" is
- **Environmental**: "This weather is terrible" → unclear what weather/location

## The Solution

The contextual grounding system automatically converts these references to absolute, context-independent forms:

- "It's my birthday today" → "My birthday is January 15, 2024"
- "It's really hot outside" → "The weather in Jakarta, Indonesia is very hot"
- "This guy is annoying" → "John (from the meeting) is annoying"

## How It Works

### 1. Context Detection
The system uses an LLM to analyze memory text and detect context-dependent references:

```python
dependencies = agent._analyze_context_dependencies("It's really hot outside")
# Returns: {"spatial": ["outside"], "environmental": ["hot"], ...}
```

### 2. Context Capture
Current context is captured from multiple sources:

```python
agent.set_context(
    location="Jakarta, Indonesia",
    activity="just arrived from flight", 
    people_present=["travel companion"],
    weather="hot and humid",
    temperature="32°C"
)
```

### 3. Contextual Grounding
References are resolved using current context:

```python
# Original: "It's really hot outside"
# Grounded: "The weather in Jakarta, Indonesia is very hot"
```

## Usage

### Setting Context

```python
# Set location and activity
agent.set_context(
    location="Jakarta, Indonesia",
    activity="business meeting"
)

# Add people and environment
agent.set_context(
    people_present=["Sarah", "Mike"],
    weather="rainy",
    mood="excited"
)
```

### Storing Memories

```python
# Automatic grounding (default)
agent.store_memory("It's really crowded here")
# Stored as: "Jakarta, Indonesia is really crowded"

# Without grounding
agent.store_memory("It's really crowded here", apply_grounding=False)
# Stored as: "It's really crowded here"
```

### CLI Commands

```bash
# Set context
context location="Jakarta, Indonesia" activity="traveling" people="John,Sarah"

# View current context  
context-info

# Store memory with grounding
remember "It's so hot outside"

# Store memory without grounding
remember-raw "It's so hot outside"
```

## Context Types Supported

### Temporal References
- **today, yesterday, now, this morning, last week**
- Resolved to: specific dates and times

### Spatial References  
- **here, outside, this place, nearby, upstairs**
- Resolved to: specific locations

### Personal References
- **this guy, my boss, the meeting, that person**
- Resolved to: specific names when determinable

### Environmental References
- **this weather, the current situation, right now**
- Resolved to: specific conditions with location/time

### Demonstrative References
- **this, that, these, those** (when unclear)
- Resolved to: specific objects/concepts when possible

## Benefits

### 1. Future-Proof Memories
Memories remain meaningful regardless of when they're retrieved:
- "Today was great" → "January 15, 2024 was great"

### 2. Location-Independent Retrieval
Spatial references work from anywhere:
- "It's hot outside" → "Jakarta, Indonesia is hot"

### 3. Better Search Results
Grounded memories match more search queries:
- Query: "weather in Jakarta" matches "It's hot outside" (grounded)

### 4. Preserved Context
Original meaning maintained while adding clarity:
- Original context preserved in metadata
- Grounded version used for search and retrieval

## Technical Details

### Storage Format
```json
{
  "raw_text": "It's really hot outside",
  "final_text": "The weather in Jakarta, Indonesia is very hot",
  "grounding_applied": true,
  "grounding_info": {
    "changes_made": [
      {
        "original": "outside", 
        "replacement": "in Jakarta, Indonesia",
        "type": "spatial"
      }
    ]
  },
  "context_snapshot": {
    "temporal": {"date": "January 15, 2024"},
    "spatial": {"location": "Jakarta, Indonesia"}
  }
}
```

### Performance
- Context analysis: ~200ms per memory
- Grounding: ~500ms per memory (only when needed)
- No impact on retrieval speed
- Embeddings generated from grounded text for better search

## Examples

### Travel Scenario
```python
# Set travel context
agent.set_context(location="Tokyo, Japan", activity="vacation")

# Store memories
agent.store_memory("The food here is amazing")
# → "The food in Tokyo, Japan is amazing"

agent.store_memory("I'm so tired from the flight today") 
# → "I'm so tired from the flight on January 15, 2024"
```

### Work Scenario  
```python
# Set work context
agent.set_context(
    location="Office, San Francisco",
    activity="team meeting",
    people_present=["Alice", "Bob", "Carol"]
)

agent.store_memory("Alice had great ideas in the meeting")
# → "Alice had great ideas in the team meeting at Office, San Francisco"
```

## Configuration

### Disable Grounding
```python
# Globally disable
agent.store_memory(text, apply_grounding=False)

# Or use CLI
remember-raw "Store this exactly as-is"
```

### Context Sources
Context can come from:
- Manual setting via `set_context()`
- GPS/location services (future)
- Calendar integration (future)
- User profiles (future)

## Best Practices

1. **Set context before storing memories** for best results
2. **Update context when changing locations/activities**
3. **Use specific locations** ("Jakarta, Indonesia" vs "here")
4. **Include relevant people** in social context
5. **Review grounded memories** to ensure accuracy

## Limitations

- Requires OpenAI API for context analysis and grounding
- May occasionally misinterpret complex references
- Context must be manually set (no automatic detection yet)
- Works best with clear, simple references

## Future Enhancements

- Automatic location detection via GPS
- Calendar integration for temporal context
- Photo/image context analysis
- Multi-language support
- Confidence scoring for grounding decisions
