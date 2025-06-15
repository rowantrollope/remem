"""
Main entry point for the LangGraph agent.
"""

import os
import sys
from dotenv import load_dotenv
from agent import LangGraphAgent

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
    """Main function to run the agent."""
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        print("You can copy .env.example to .env and fill in your API key.")
        sys.exit(1)
    
    print("Initializing LangGraph Agent...")
    
    try:
        # Create the agent
        agent = LangGraphAgent()
        
        # Check if we have command line arguments
        if len(sys.argv) > 1:
            # Run with command line input
            user_input = " ".join(sys.argv[1:])
            print(f"User: {user_input}")
            response = agent.run(user_input)
            print(f"Agent: {response}")
        else:
            # Start interactive chat
            agent.chat()
            
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure you have installed all dependencies with: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
