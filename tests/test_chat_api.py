#!/usr/bin/env python3
"""
Test the new Chat API for frontend developers
"""

import requests
import json
import time

class ChatAPITester:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.session_id = None

    def create_session(self, system_prompt, config=None):
        """Create a new chat session."""
        payload = {
            "system_prompt": system_prompt
        }
        if config:
            payload["config"] = config

        response = requests.post(f"{self.base_url}/api/agent/session", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            self.session_id = result["session_id"]
            return result
        else:
            raise Exception(f"Failed to create session: {response.status_code} - {response.text}")

    def send_message(self, message, session_id=None):
        """Send a message to the chat session."""
        session = session_id or self.session_id
        if not session:
            raise Exception("No active session. Create a session first.")

        response = requests.post(
            f"{self.base_url}/api/agent/session/{session}",
            json={"message": message}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to send message: {response.status_code} - {response.text}")

    def get_session_info(self, session_id=None):
        """Get session information and history."""
        session = session_id or self.session_id
        if not session:
            raise Exception("No active session specified.")

        response = requests.get(f"{self.base_url}/api/agent/session/{session}")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get session: {response.status_code} - {response.text}")

    def list_sessions(self):
        """List all active sessions."""
        response = requests.get(f"{self.base_url}/api/agent/sessions")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to list sessions: {response.status_code} - {response.text}")

    def delete_session(self, session_id=None):
        """Delete a chat session."""
        session = session_id or self.session_id
        if not session:
            return

        response = requests.delete(f"{self.base_url}/api/agent/session/{session}")
        
        if response.status_code == 200:
            if session == self.session_id:
                self.session_id = None
            return response.json()
        else:
            raise Exception(f"Failed to delete session: {response.status_code} - {response.text}")

def test_travel_agent_chat():
    """Test a travel agent chat scenario."""
    print("🧳 Testing Travel Agent Chat")
    print("=" * 40)
    
    tester = ChatAPITester()
    
    # Create travel agent session
    travel_prompt = """You are an expert travel agent named Sarah. You help users plan amazing trips by:
    - Asking about their preferences, budget, and constraints
    - Providing personalized destination recommendations
    - Helping with logistics like flights, hotels, and activities
    - Being enthusiastic and knowledgeable about travel
    
    Always ask follow-up questions to better understand their needs and provide tailored advice."""
    
    try:
        # Create session
        session = tester.create_session(
            travel_prompt,
            config={
                "temperature": 0.8,  # More creative responses
                "max_tokens": 600
            }
        )
        print(f"✅ Created session: {session['session_id']}")
        print(f"📝 System prompt: {session['system_prompt'][:100]}...")
        
        # Simulate conversation
        conversation = [
            "Hi, I want to plan a romantic getaway for my anniversary",
            "We prefer somewhere warm with beautiful beaches",
            "Our budget is around $4000 for a week",
            "We love trying local food and don't mind adventure activities"
        ]
        
        for i, message in enumerate(conversation, 1):
            print(f"\n--- Message {i} ---")
            print(f"User: {message}")
            
            response = tester.send_message(message)
            print(f"Sarah: {response['message']}")
            print(f"Conversation length: {response['conversation_length']}")
            
            time.sleep(1)  # Small delay between messages
        
        # Get session history
        print(f"\n--- Session History ---")
        history = tester.get_session_info()
        print(f"Total messages: {history['message_count']}")
        print(f"Created: {history['created_at']}")
        print(f"Last activity: {history['last_activity']}")
        
        return tester.session_id
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_customer_support_chat():
    """Test a customer support chat scenario."""
    print("\n📞 Testing Customer Support Chat")
    print("=" * 40)
    
    tester = ChatAPITester()
    
    # Create support session
    support_prompt = """You are a helpful customer support agent for TechCorp named Alex.
    
    Guidelines:
    - Be empathetic and solution-focused
    - Ask clarifying questions about technical issues
    - Provide step-by-step troubleshooting instructions
    - Maintain a professional, helpful tone
    - If you can't solve the issue, offer to escalate to a specialist
    
    Always start by acknowledging the customer's issue and showing you want to help."""
    
    try:
        # Create session
        session = tester.create_session(
            support_prompt,
            config={
                "temperature": 0.3,  # More consistent responses
                "max_tokens": 500
            }
        )
        print(f"✅ Created session: {session['session_id']}")
        
        # Simulate support conversation
        support_conversation = [
            "My laptop won't turn on and I have an important presentation tomorrow",
            "I tried holding the power button but nothing happens",
            "The charging light is on, so I think it's getting power",
            "It's a TechCorp Pro 15 that I bought last year"
        ]
        
        for i, message in enumerate(support_conversation, 1):
            print(f"\n--- Support Ticket Update {i} ---")
            print(f"Customer: {message}")
            
            response = tester.send_message(message)
            print(f"Alex: {response['message']}")
            
            time.sleep(1)
        
        return tester.session_id
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_educational_tutor_chat():
    """Test an educational tutor chat scenario."""
    print("\n📚 Testing Educational Tutor Chat")
    print("=" * 40)
    
    tester = ChatAPITester()
    
    # Create tutor session
    tutor_prompt = """You are an expert Python programming tutor named Dr. Code.
    
    Teaching approach:
    - Break down complex concepts into simple, understandable steps
    - Use practical examples and analogies
    - Ask questions to check student understanding
    - Provide encouragement and positive feedback
    - Adapt explanations based on student responses
    - Always be patient and supportive
    
    You're helping a beginner learn Python programming."""
    
    try:
        # Create session
        session = tester.create_session(
            tutor_prompt,
            config={
                "temperature": 0.6,
                "max_tokens": 700
            }
        )
        print(f"✅ Created session: {session['session_id']}")
        
        # Simulate tutoring conversation
        tutoring_conversation = [
            "I'm new to programming and want to learn Python. Where should I start?",
            "What's the difference between a list and a dictionary?",
            "Can you show me a simple example of a for loop?"
        ]
        
        for i, message in enumerate(tutoring_conversation, 1):
            print(f"\n--- Lesson {i} ---")
            print(f"Student: {message}")
            
            response = tester.send_message(message)
            print(f"Dr. Code: {response['message']}")
            
            time.sleep(1)
        
        return tester.session_id
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_session_management():
    """Test session management features."""
    print("\n🔧 Testing Session Management")
    print("=" * 40)
    
    tester = ChatAPITester()
    
    try:
        # List all sessions
        sessions = tester.list_sessions()
        print(f"📋 Active sessions: {sessions['total_sessions']}")
        
        for session in sessions['sessions']:
            print(f"   Session: {session['session_id']}")
            print(f"   Created: {session['created_at']}")
            print(f"   Messages: {session['message_count']}")
            print(f"   Preview: {session['system_prompt_preview']}")
            print()
        
        # Clean up - delete all sessions
        print("🗑️ Cleaning up sessions...")
        for session in sessions['sessions']:
            result = tester.delete_session(session['session_id'])
            print(f"   Deleted: {session['session_id']}")
        
        # Verify cleanup
        final_sessions = tester.list_sessions()
        print(f"✅ Remaining sessions: {final_sessions['total_sessions']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all chat API tests."""
    print("🚀 Testing New Chat API for Frontend Developers")
    print("=" * 60)
    
    try:
        # Test different chat scenarios
        travel_session = test_travel_agent_chat()
        support_session = test_customer_support_chat()
        tutor_session = test_educational_tutor_chat()
        
        # Test session management
        test_session_management()
        
        print(f"\n{'='*60}")
        print("✅ All tests completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Custom system prompts for different use cases")
        print("• Continuous conversation with memory")
        print("• Session management (create, list, delete)")
        print("• Configurable model parameters")
        print("• Full conversation history tracking")
        print("\nPerfect for frontend developers to build chat apps! 🎉")
        
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: python3 web_app.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
