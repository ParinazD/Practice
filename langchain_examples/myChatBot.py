import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Define the Model (OpenRouter Configuration)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def get_openrouter_llm(max_tokens: int = 512, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        # Using a model that supports tool calling (like gpt-4o-mini)
        model="openai/gpt-4o-mini", 
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        max_tokens=max_tokens,
        temperature=temperature
    )

# Define Tools

@tool
def get_word_count(text: str) -> int:
    """Calculates the exact number of words in a given piece of text."""
    return len(text.split())

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b


# Main Agent Loop
def main():
    # Initialize OpenRouter model
    model = get_openrouter_llm()
    
    # Pack tools into a list
    tools = [get_word_count, multiply_numbers]
    
    # System prompt: Strictly tells the agent to be concise and use tools
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are a helpful AI assistant. You must answer questions very concisely. "
            "Always use your provided tools to answer questions whenever they are applicable."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(model, tools, prompt_template)
    
    # Executor runs the agent, calls the tools, and returns the result
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # chat history memory
    history = ChatMessageHistory()
    
    print("AI: Hello! I am ready. I have tools to count words and multiply numbers. (Type 'quit' to exit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        if not user_input.strip():
            continue
            
        # Run the agent through the executor
        response = agent_executor.invoke({
            "chat_history": history.messages,
            "input": user_input
        })
        
        # Add user input and final agent response to memory
        history.add_user_message(user_input)
        history.add_ai_message(response["output"])
        
        print(f"AI: {response['output']}\n")

if __name__ == "__main__":
    main()