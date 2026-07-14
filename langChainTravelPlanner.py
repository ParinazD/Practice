import os
import random
from typing import List
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub

# 1. OPENROUTER CONFIGURATION
# OpenRouter uses the OpenAI client format.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemma-4-26b-a4b-it:free"

# Initialize with OpenRouter's base URL and config
llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model=MODEL_NAME,
    temperature=0.7
)

# 2. DEFINING TOOLS (FUNCTIONS)

def calculate_budget(total_budget: int, days: int) -> str:
    """Calculates a rough daily budget breakdown for the trip."""
    daily_allowance = total_budget // days # integer division
    result = f"Total Budget: ${total_budget} for {days} days. Your daily budget is ${daily_allowance} per day for food and activities."
    print(f"[Tool Executed] calculate_budget -> {result}")
    return result

def arrange_flights(destination: str) -> str:
    result = f"Mock Flight Booking: Found a round-trip ticket to {destination} at $450. Successfully reserved!"
    print(f"[Tool Executed] arrange_flights -> {result}")
    return result

def generate_itinerary(destination: str, activity_type: str, days: int) -> str:
    """Generates a randomized itinerary based on user preference + destination."""
    # Base generic events that can happen anywhere
    events = ["have a picnic", "eat at local restaurant", "go sight-seeing"]
    
    # Add a region/activity specific event based on what the user wants to do
    if "hike" in activity_type.lower():
        events.append(f"hike trails of {destination}")
    elif "beach" in activity_type.lower():
        events.append(f"relax beach in {destination}")
    else:
        events.append(f"explore local places in {destination}")
        
    itinerary_output = f"- {days}-Day Itinerary for {destination} ({activity_type}) ---\n"
    
    # Loop through each day and pick a random order of activities
    for day in range(1, days + 1):
        # random.sample shuffles and picks unique items so the day feels varied
        todays_activities = random.sample(events, k=3) 
        itinerary_output += f"Day {day}:\n"
        itinerary_output += f"  - Morning: {todays_activities[0]}\n"
        itinerary_output += f"  - Afternoon: {todays_activities[1]}\n"
        itinerary_output += f"  - Evening: {todays_activities[2]}\n"
        
    print(f"[Tool Executed] generate_itinerary -> Created a {days}-day plan.")
    return itinerary_output


# 3. WRAPPING FUNCTIONS AS LANGCHAIN TOOLS
# StructuredTool.from_function automatically uses the Python function's docstring 
# to tell the LLM what the tool does and what arguments it expects.
tools = [
    StructuredTool.from_function(calculate_budget),
    StructuredTool.from_function(arrange_flights),
    StructuredTool.from_function(generate_itinerary)
]


# 4. SETTING UP AGENT
# We pull a standard system prompt template from LangChain's public repository
# This prompt tells the agent how to think and use tools effectively.
#prompt = hub.pull("hwchase17/openai-functions-agent")

# Instead of hub.pull(), construct a proper template manually:
prompt = ChatPromptTemplate.from_messages([
    # 1. System message: Set the persona and rules
    ("system", "You are a helpful, simple travel agent. Always use your tools to calculate budgets, flights, and itineraries."),
    
    # 2. User input: Where the user query ("I want to go to Hawaii...") goes
    ("human", "{input}"),
    
    # 3. Mandatary placeholder: LangChain tracks the agent's tool-using history.
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Construct the OpenAI Functions agent using our LLM, Tools, and Prompt setup
agent = create_openai_functions_agent(llm, tools, prompt)

# The AgentExecutor is the runtime that actually calls the tools when the LLM asks for them
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 5. RUNNING THE PROGRAM
if __name__ == "__main__":
    # The beginner-friendly user request containing all necessary info
    user_query = "I want to go to Hawaii for a beach trip. I have a budget of $2000 and want to stay for 4 days. Please plan everything."
    
    print("Starting Travel Planner Agent...\n")
    
    # Run the agent! It will read the query, figure out it needs to call all 3 tools, and compile a final answer.
    response = agent_executor.invoke({"input": user_query})
    
    print("\nFINAL RESPONSE")
    print(response["output"])