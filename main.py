"""
Main entry point for the LangGraph Memory Agent.
"""

import os
import sys
from dotenv import load_dotenv
from langgraph_memory_agent import LangGraphMemoryAgent

import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Make sure this import is here
from memory_agent import MemoryAgent

app = Flask(__name__)

# Add CORS with explicit configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:3001"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

def main():
    """Main function to run the memory agent."""
    # Load environment variables
    load_dotenv()

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        print("You can copy .env.example to .env and fill in your API key.")
        sys.exit(1)

    print("Initializing LangGraph Memory Agent...")
    try:
        # Create the memory agent
        agent = LangGraphMemoryAgent()
    except Exception as e:
        print(f"Error initializing memory agent: {e}")
        print("Make sure Redis is running and all dependencies are installed.")
        sys.exit(1)

    try:
        # Check if we have command line arguments
        if len(sys.argv) > 1:
            # Run with command line input
            user_input = " ".join(sys.argv[1:])
            print(f"User: {user_input}")
            response = agent.run(user_input)
            print(f"Agent: {response}")
        else:
            # Start interactive chat
            print("🧠 Memory Agent - You can store and retrieve memories!")
            print("Examples:")
            print("- 'Remember that I like pizza'")
            print("- 'What do I like to eat?'")
            print("- 'Store this: I met John at the coffee shop'")
            print("- 'Set context: I'm at home working on my laptop'")
            agent.chat()

    except Exception as e:
        print(f"Error running memory agent: {e}")
        print("Make sure you have installed all dependencies with: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
