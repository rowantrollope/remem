# Memory Agent Web UI

A minimalist web interface for the Memory Agent application.

## Features

- **Clean, Modern Design**: Minimalist interface with gradient backgrounds and smooth animations
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Five Main Functions**:
  - 💾 **Store Memory**: Add new memories with automatic tag extraction
  - 🔍 **Search Memories**: Find relevant memories using vector similarity
  - 🤔 **Ask Questions**: Get AI-powered answers based on stored memories
  - 🗑️ **Delete Memories**: Remove specific memories by ID with confirmation
  - 📊 **Memory Statistics**: View comprehensive information about stored memories

## Quick Start

1. **Start the web server**:
   ```bash
   source venv/bin/activate
   python web_app.py
   ```

2. **Open your browser** and go to: `http://localhost:5001`

3. **Start using the interface**:
   - Store memories by typing in the text area and clicking "Remember"
   - Search for memories by entering a query and clicking "Search"
   - Ask questions about your memories and get intelligent answers

## API Endpoints

The web UI uses these REST API endpoints:

- `POST /api/remember` - Store a new memory
- `POST /api/recall` - Search for memories
- `POST /api/ask` - Ask a question
- `DELETE /api/delete/<memory_id>` - Delete a specific memory
- `GET /api/memory-info` - Get comprehensive memory statistics
- `GET /api/status` - Check system status

## Keyboard Shortcuts

- **Store Memory**: `Ctrl + Enter` in the memory text area
- **Search**: `Enter` in the search input
- **Ask Question**: `Enter` in the question input

## Design Features

- **Gradient Backgrounds**: Beautiful purple-blue gradients
- **Smooth Animations**: Hover effects and loading states
- **Visual Feedback**: Success, error, and loading indicators
- **Memory Cards**: Clean display of search results with similarity scores
- **Confidence Indicators**: AI answers include appropriate confidence levels

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Styling**: Modern CSS with gradients and animations
- **Memory Engine**: Redis VectorSet with OpenAI embeddings

## Mobile Support

The interface is fully responsive and works well on:
- Desktop computers
- Tablets
- Mobile phones

## Error Handling

The UI provides clear feedback for:
- Network errors
- Invalid inputs
- System failures
- Empty results

Enjoy using your Memory Agent! 🧠✨
