# Remember - AI-Powered Memory Assistant

A Next.js web application that provides a beautiful interface for the Memory Agent API, allowing you to store, search, and query your personal memories using AI.

## Features

- **Save Memories**: Store important moments and information
- **Ask Questions**: Query your memories with natural language
- **Search Memories**: Find specific memories using semantic search
- **Real-time Stats**: Track your memory collection, conversations, and system information
- **System Monitoring**: View detailed information about the vector database and embedding model
- **Beautiful UI**: Modern, responsive interface built with Tailwind CSS and shadcn/ui

## Prerequisites

Before running this application, you need to have the Memory Agent API server running. The API should be available at `http://localhost:5001` by default.

### Memory Agent API

The application expects a Flask API server with the following endpoints (all require vectorset name):
- `POST /api/memory/{vectorset_name}` - Store a new memory
- `POST /api/memory/{vectorset_name}/ask` - Ask a question about memories
- `POST /api/memory/{vectorset_name}/search` - Search memories
- `GET /api/status` - Check API status
- `GET /api/memory/{vectorset_name}` - Get detailed memory system information

**Note:** All memory endpoints now require an explicit vectorset name (e.g., "memories") in the URL path.

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd remember
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure the API URL** (optional):
   ```bash
   cp .env.local.example .env.local
   ```
   Edit `.env.local` to set your Memory Agent API URL if it's different from the default:
   ```
   NEXT_PUBLIC_API_BASE_URL=http://localhost:5001
   ```

4. **Start the Memory Agent API server**:
   Make sure your Memory Agent Flask server is running on port 5001 (or the port you configured).

5. **Run the development server**:
   ```bash
   npm run dev
   ```

6. **Open the application**:
   Navigate to [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

### Saving Memories
1. Click on the "Save" tab
2. Type what you want to remember (e.g., "I had lunch with John at the Italian restaurant")
3. Press Enter or click the save button

### Asking Questions
1. Click on the "Ask" tab
2. Ask a natural language question about your memories (e.g., "When did I last eat Italian food?")
3. The AI will search your memories and provide an answer

### Searching Memories
1. Click on the "Recall" tab
2. Enter search terms to find relevant memories
3. View the matching memories with their timestamps and metadata

## API Integration

The application integrates with the Memory Agent API through the following client (`src/lib/api.ts`):

- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Loading States**: Visual feedback during API calls
- **Status Monitoring**: Real-time API status checking
- **Type Safety**: Full TypeScript support for API responses

## Development

### Project Structure
```
src/
├── app/
│   ├── page.tsx          # Main application component
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Global styles
├── lib/
│   ├── api.ts           # Memory Agent API client
│   └── utils.ts         # Utility functions
└── components/
    └── ui/              # shadcn/ui components
```

### Technologies Used
- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components
- **Lucide React** - Icons

## Troubleshooting

### API Connection Issues
- Ensure the Memory Agent API server is running
- Check that the API URL in `.env.local` is correct
- Verify the API server is accessible from your browser

### Common Errors
- **"Memory agent not initialized"**: The API server is not running or not properly configured
- **"Failed to connect to Memory Agent API"**: Network connectivity issues or wrong API URL

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
