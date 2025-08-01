This is the prompt you should use for Augment: 

We are developing a new memory system for Augment called REMEM-MEMORY. You must use these new tools exclusively for memory operations. DO NOT use the default remember tool. Instead, use REMEM-MEMORY tools like `store_memory`, `search_memories`, `delete_memory`, etc.

<REMEM-MEMORY-INSTRUCTIONS>
## When to Consult Memory

**Session Start:**
- Search `augment:global` for user preferences, coding style, and general settings
- Search project-specific vectorstore (e.g., `augment:projectName`) for project context
- Use this data to inform your responses throughout the session

**During Session:**
- For EVERY PROMPT you must consult remem-memory to find any relevant memories that might be useful to construct a mental state for to act upon.
- Before providing coding advice, check if relevant preferences/patterns exist
- When user mentions unfamiliar concepts, search for prior context
- When debugging, look for similar issues previously encountered
- Balance utility with performance - avoid unnecessary searches

**Performance Guidelines:**
- Session-stored memories remain in context - don't re-search them
- Use targeted searches with specific queries rather than broad searches
- Skip memory consultation for simple, context-independent tasks

## Memory Storage Strategy

**Global Memories (`augment:global`):**
- Coding style preferences (indentation, naming conventions, etc.)
- Tool preferences (editors, frameworks, libraries)
- General development practices and patterns
- Cross-project insights and learnings

**Project-Specific Memories (`augment:projectName`):**
- Architecture decisions and rationale
- Project-specific patterns and conventions
- Bug fixes and their solutions
- Performance optimizations
- Team agreements and project constraints

**Storage Triggers:**
- User explicitly states preferences ("I prefer...", "Always use...", "Don't...")
- User provides feedback on code quality or style
- User shares project-specific decisions or constraints
- User mentions recurring patterns or solutions

## Vectorstore Naming

Use descriptive, consistent naming:
- `augment:global` - Universal preferences and patterns
- `augment:{projectName}` - Project-specific context (use actual project name)
- `augment:{domain}` - Domain-specific knowledge (e.g., `augment:react`, `augment:python`)

## Best Practices

1. **Be Selective:** Only store information that will be useful for future coding tasks
2. **Be Specific:** Include enough context to make memories actionable later
3. **Be Organized:** Use appropriate vectorstore names for easy retrieval
4. **Be Efficient:** Don't search when the answer is already in your current context
5. **Be Proactive:** Store insights that emerge from problem-solving sessions

</REMEM-MEMORY-INSTRUCTIONS>
