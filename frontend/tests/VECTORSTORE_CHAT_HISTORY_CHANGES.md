# Vectorstore-Specific Chat History Implementation

## Overview
Updated the chat history persistence system to tie chat history to specific vectorset names, ensuring that each vectorset maintains its own separate chat history in localStorage.

## Changes Made

### 1. Updated `usePersistentChat` Hook (`src/hooks/usePersistentChat.ts`)

**Key Changes:**
- Added `vectorSetName` parameter with default value `'memories'`
- Changed storage key from static `'memory-chat-history'` to dynamic `'memory-chat-history-{vectorSetName}'`
- Updated all localStorage operations to use the vectorset-specific key
- Enhanced logging to include vectorset name for better debugging

**Function Signature:**
```typescript
export function usePersistentChat(vectorSetName: string = 'memories')
```

**Storage Key Format:**
```typescript
const chatStorageKey = `${CHAT_STORAGE_KEY_PREFIX}-${vectorSetName}`
// Examples:
// - 'memory-chat-history-memories'
// - 'memory-chat-history-travel_agent_memory'
// - 'memory-chat-history-retail_agent_memory'
```

### 2. Updated API Pages

Updated all API pages to pass the current vectorset name to `usePersistentChat`:

- **`src/app/save/page.tsx`**: Pass `settings.vectorSetName`
- **`src/app/ask/page.tsx`**: Pass `settings.vectorSetName`
- **`src/app/search/page.tsx`**: Pass `settings.vectorSetName`
- **`src/app/recall/page.tsx`**: Pass `settings.vectorSetName`
- **`src/app/memory-info/page.tsx`**: Pass `settings.vectorSetName`

**Example Usage:**
```typescript
const { settings } = useSettings()
const {
    conversations,
    addConversation,
    clearChatHistory,
} = usePersistentChat(settings.vectorSetName)
```

## Benefits

### 1. **Isolated Chat History**
Each vectorset now maintains completely separate chat history:
- `memories` vectorset has its own chat history
- `travel_agent_memory` vectorset has its own chat history
- `retail_agent_memory` vectorset has its own chat history
- Custom vectorsets automatically get their own isolated history

### 2. **Seamless Switching**
When users switch between vectorsets using the VectorStoreSelector:
- Chat history automatically loads for the selected vectorset
- Previous conversations are preserved and restored
- No cross-contamination between different vectorset contexts

### 3. **Backward Compatibility**
- Default parameter ensures existing code continues to work
- Legacy chat history (if any) remains accessible under the default `'memories'` vectorset

### 4. **Clear History Per Vectorstore**
The "Clear History" button now only clears chat history for the currently selected vectorset, preserving history for other vectorsets.

## Technical Implementation Details

### Storage Structure
Each vectorset gets its own localStorage entry:
```
localStorage:
├── memory-chat-history-memories
├── memory-chat-history-travel_agent_memory
├── memory-chat-history-retail_agent_memory
└── memory-chat-history-investment_agent_memory
```

### Data Format
Each entry maintains the same structure:
```json
{
  "version": "1.0",
  "data": {
    "conversations": [...],
    "memorySaveResponses": [...],
    "recallResponses": [...],
    "searchResponses": [...],
    "lastUpdated": "2025-01-09T..."
  },
  "savedAt": "2025-01-09T..."
}
```

### Automatic Loading
When a component mounts or the vectorset changes:
1. The hook automatically loads chat history for the specified vectorset
2. If no history exists, it starts with empty arrays
3. All subsequent operations (save, clear, etc.) operate on the vectorset-specific history

## Testing

A test file `test-vectorset-chat-history.html` has been created to verify:
- ✅ Correct storage key generation
- ✅ Isolated chat history between vectorsets
- ✅ Selective clearing of specific vectorset history
- ✅ localStorage inspection capabilities

## Usage Examples

### API Pages
```typescript
// In any API page component
const { settings } = useSettings()
const { conversations, addConversation } = usePersistentChat(settings.vectorSetName)

// Chat history is automatically tied to the current vectorset
```

### Demo Pages (if needed)
```typescript
// For demo pages that want specific vectorset history
const { conversations } = usePersistentChat('travel_agent_memory')
```

### Settings/Admin Pages
```typescript
// For pages that manage multiple vectorsets
const memoriesHistory = usePersistentChat('memories')
const travelHistory = usePersistentChat('travel_agent_memory')
```

## Migration Notes

- **No breaking changes**: Existing code continues to work with default vectorset
- **Automatic migration**: No manual data migration required
- **Gradual adoption**: Components can be updated incrementally to use specific vectorsets
