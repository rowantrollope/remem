#!/usr/bin/env python3
"""
Calendar Memory Assistant Demo

Demonstrates how memory transforms a basic calendar assistant into a powerful personal agent.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import the memory agent
from memory_agent import MemoryAgent

class CalendarAssistant:
    """Calendar assistant that demonstrates the power of memory for agentic applications."""
    
    def __init__(self, use_memory: bool = True):
        """Initialize the calendar assistant.
        
        Args:
            use_memory: Whether to use memory capabilities
        """
        self.use_memory = use_memory
        self.memory_agent = MemoryAgent() if use_memory else None
        
        # Simple in-memory calendar for demo purposes
        self.calendar = []
    
    def add_event(self, event_text: str) -> Dict[str, Any]:
        """Add an event to the calendar.
        
        Args:
            event_text: Description of the event
            
        Returns:
            Response with event details
        """
        # Basic event parsing (very limited without memory)
        event = self._parse_event(event_text)
        
        # Store in calendar
        self.calendar.append(event)
        
        # If using memory, store rich context about the event
        if self.use_memory and self.memory_agent:
            memory_text = f"Added calendar event: {event['title']} on {event['date']} at {event['time']}."
            if event['people']:
                memory_text += f" With: {', '.join(event['people'])}."
            if event['location']:
                memory_text += f" Location: {event['location']}."
            if event['notes']:
                memory_text += f" Notes: {event['notes']}."
                
            self.memory_agent.store_memory(memory_text)
            
            # Store additional context about people involved
            for person in event['people']:
                context_memory = f"I have a meeting with {person} about {event['title']} on {event['date']}."
                self.memory_agent.store_memory(context_memory)
        
        return {"status": "success", "event": event}
    
    def _parse_event(self, event_text: str) -> Dict[str, Any]:
        """Parse event text into structured data.
        
        Args:
            event_text: Raw event description
            
        Returns:
            Structured event data
        """
        # Without memory, we have very basic parsing capabilities
        if not self.use_memory:
            # Very simplistic parsing (error-prone)
            parts = event_text.split(" on ")
            title = parts[0].strip()
            date_time = parts[1].strip() if len(parts) > 1 else "tomorrow"
            
            return {
                "title": title,
                "date": date_time,
                "time": "9:00 AM",  # Default
                "people": [],
                "location": "",
                "notes": ""
            }
        
        # With memory, we can use the LLM for sophisticated parsing
        # and leverage past context about people, locations, etc.
        else:
            # Use memory agent to analyze the event with context
            prompt = f"""
            Parse the following calendar event text into structured data:
            "{event_text}"
            
            Extract:
            - Event title/purpose
            - Date (resolve relative dates like "tomorrow", "next Tuesday")
            - Time
            - People involved
            - Location
            - Any notes or additional details
            
            Return as JSON.
            """
            
            # In a real implementation, we would call the LLM here
            # For demo purposes, we'll simulate a better result
            today = datetime.now()
            tomorrow = today + timedelta(days=1)
            
            # Simulate better parsing with memory context
            if "Sarah" in event_text:
                # Memory knows Sarah is a project manager
                people = ["Sarah Johnson"]
                notes = "Sarah is the project manager for the Redis project"
            elif "team" in event_text.lower():
                # Memory knows team members
                people = ["Alex", "Jamie", "Taylor"]
                notes = "Weekly team sync"
            else:
                people = []
                notes = ""
                
            if "coffee" in event_text.lower():
                # Memory knows preferred coffee shop
                location = "Starbucks downtown"
            elif "office" in event_text.lower():
                # Memory knows office location
                location = "Main office, Conference Room B"
            else:
                location = ""
                
            return {
                "title": event_text.split(" on ")[0] if " on " in event_text else event_text,
                "date": tomorrow.strftime("%Y-%m-%d"),
                "time": "10:00 AM",
                "people": people,
                "location": location,
                "notes": notes
            }
    
    def query_calendar(self, query: str) -> str:
        """Query the calendar with natural language.
        
        Args:
            query: Natural language query about calendar
            
        Returns:
            Response to the query
        """
        if not self.use_memory:
            # Without memory, we can only do basic keyword matching
            if "next" in query and "meeting" in query:
                if not self.calendar:
                    return "You don't have any upcoming meetings."
                return f"Your next meeting is {self.calendar[0]['title']} on {self.calendar[0]['date']}."
            elif "today" in query:
                today_events = [e for e in self.calendar if "today" in e['date'].lower()]
                if not today_events:
                    return "You don't have any events scheduled for today."
                return f"Today you have: {today_events[0]['title']}."
            else:
                return "I'm not sure how to answer that question about your calendar."
        else:
            # With memory, we can leverage the memory agent for sophisticated queries
            # that understand context, people, and past interactions
            if self.memory_agent:
                # First search memories for relevant context
                memories = self.memory_agent.search_memories(query, top_k=5)
                
                # Then use the memory agent to answer the question
                answer = self.memory_agent.answer_question(query)
                
                if answer['confidence'] == 'low':
                    # Fall back to basic calendar search
                    if "meeting with" in query.lower():
                        person = query.lower().split("meeting with")[1].strip().split()[0]
                        matching_events = [e for e in self.calendar if any(person.lower() in p.lower() for p in e['people'])]
                        
                        if matching_events:
                            event = matching_events[0]
                            return f"You have a meeting with {person} on {event['date']} at {event['time']} about {event['title']}."
                        else:
                            return f"I don't see any meetings scheduled with {person}."
                    
                    # Handle other query types with memory context
                    if "next week" in query.lower():
                        # Memory helps understand what "next week" means relative to today
                        today = datetime.now()
                        next_week_start = today + timedelta(days=(7-today.weekday()))
                        next_week_end = next_week_start + timedelta(days=6)
                        
                        next_week_events = []
                        for event in self.calendar:
                            try:
                                event_date = datetime.strptime(event['date'], "%Y-%m-%d")
                                if next_week_start <= event_date <= next_week_end:
                                    next_week_events.append(event)
                            except:
                                # Handle dates that don't match expected format
                                pass
                        
                        if next_week_events:
                            events_text = "\n".join([f"- {e['title']} on {e['date']} at {e['time']}" for e in next_week_events])
                            return f"Next week you have:\n{events_text}"
                        else:
                            return "You don't have any events scheduled for next week."
                
                return answer['answer']
            
            return "Memory agent not available."

def run_demo():
    """Run the calendar assistant demo."""
    print("=" * 50)
    print("Calendar Assistant Demo - Memory vs. No Memory")
    print("=" * 50)
    
    # Create assistants
    basic_assistant = CalendarAssistant(use_memory=False)
    memory_assistant = CalendarAssistant(use_memory=True)
    
    # Demo scenario
    print("\n🔵 SCENARIO 1: Adding calendar events\n")
    
    # Test event 1
    event1 = "Team meeting on next Tuesday"
    
    print("👤 User: Add this to my calendar: " + event1)
    print("\n🤖 Basic Assistant (No Memory):")
    basic_result = basic_assistant.add_event(event1)
    print(f"Added: {basic_result['event']['title']} on {basic_result['event']['date']}")
    
    print("\n🧠 Memory-Enhanced Assistant:")
    memory_result = memory_assistant.add_event(event1)
    print(f"Added: {memory_result['event']['title']} on {memory_result['event']['date']} at {memory_result['event']['time']}")
    print(f"People: {', '.join(memory_result['event']['people']) if memory_result['event']['people'] else 'None specified'}")
    print(f"Location: {memory_result['event']['location'] if memory_result['event']['location'] else 'None specified'}")
    print(f"Notes: {memory_result['event']['notes'] if memory_result['event']['notes'] else 'None'}")
    
    # Test event 2
    event2 = "Coffee with Sarah to discuss the Redis project"
    
    print("\n👤 User: Schedule " + event2)
    print("\n🤖 Basic Assistant (No Memory):")
    basic_result = basic_assistant.add_event(event2)
    print(f"Added: {basic_result['event']['title']} on {basic_result['event']['date']}")
    
    print("\n🧠 Memory-Enhanced Assistant:")
    memory_result = memory_assistant.add_event(event2)
    print(f"Added: {memory_result['event']['title']} on {memory_result['event']['date']} at {memory_result['event']['time']}")
    print(f"People: {', '.join(memory_result['event']['people'])}")
    print(f"Location: {memory_result['event']['location']}")
    print(f"Notes: {memory_result['event']['notes']}")
    
    # Demo queries
    print("\n🔵 SCENARIO 2: Querying calendar information\n")
    
    # Query 1
    query1 = "When is my next meeting?"
    
    print("👤 User: " + query1)
    print("\n🤖 Basic Assistant (No Memory):")
    print(basic_assistant.query_calendar(query1))
    
    print("\n🧠 Memory-Enhanced Assistant:")
    print(memory_assistant.query_calendar(query1))
    
    # Query 2
    query2 = "What's my meeting with Sarah about?"
    
    print("\n👤 User: " + query2)
    print("\n🤖 Basic Assistant (No Memory):")
    print(basic_assistant.query_calendar(query2))
    
    print("\n🧠 Memory-Enhanced Assistant:")
    print(memory_assistant.query_calendar(query2))
    
    # Query 3
    query3 = "What do I have scheduled for next week?"
    
    print("\n👤 User: " + query3)
    print("\n🤖 Basic Assistant (No Memory):")
    print(basic_assistant.query_calendar(query3))
    
    print("\n🧠 Memory-Enhanced Assistant:")
    print(memory_assistant.query_calendar(query3))
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)

if __name__ == "__main__":
    run_demo()