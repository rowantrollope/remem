#!/bin/bash

# Quick comparison script for testing both API approaches
BASE_URL="http://localhost:5001"
SYSTEM_PROMPT="You are a helpful travel assistant. Help users plan trips and remember their preferences."
TEST_MESSAGE="I prefer window seats when flying and my wife is vegetarian"

echo "🔄 API Comparison Test"
echo "====================="
echo "System Prompt: $SYSTEM_PROMPT"
echo "Test Message: $TEST_MESSAGE"
echo ""

echo "1️⃣ Testing Enhanced /api/agent/chat"
echo "-----------------------------------"
curl -s -X POST "$BASE_URL/api/agent/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"$TEST_MESSAGE\",
    \"system_prompt\": \"$SYSTEM_PROMPT\"
  }" | jq '{
    success: .success,
    system_prompt_used: .system_prompt_used,
    response: .response[:100] + "..."
  }'

echo ""
echo "2️⃣ Testing Session-based /api/agent/session"
echo "-------------------------------------------"

# Create session
SESSION_RESPONSE=$(curl -s -X POST "$BASE_URL/api/agent/session" \
  -H "Content-Type: application/json" \
  -d "{
    \"system_prompt\": \"$SYSTEM_PROMPT\",
    \"config\": {\"use_memory\": true}
  }")

SESSION_ID=$(echo "$SESSION_RESPONSE" | jq -r '.session_id')
echo "Created session: ${SESSION_ID:0:8}..."

# Send message
curl -s -X POST "$BASE_URL/api/agent/session/$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"$TEST_MESSAGE\"}" | jq '{
    success: .success,
    conversation_length: .conversation_length,
    memory_context_included: (if .memory_context then true else false end),
    response: .message[:100] + "..."
  }'

echo ""
echo "✅ Comparison complete!"
echo ""
echo "Key differences:"
echo "• Enhanced /api/agent/chat: Stateless, LangGraph architecture"
echo "• Session-based: Stateful, custom filtering via _handle_memory_enabled_message"
