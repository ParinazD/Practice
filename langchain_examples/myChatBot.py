import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Define the Model (OpenRouter Configuration)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def get_openrouter_llm(max_tokens: int = 512, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model="openai/gpt-4o-mini", #  OpenRouter model identifier
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        max_tokens=max_tokens,
        temperature=temperature
    )

def main():
    # Initialize the model using OpenRouter function
    model = get_openrouter_llm()
    
    print("AI: Hello! How can I help you today? (Type 'quit' to exit)")
    
    #Chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        if not user_input.strip():
            continue
            
        # Invoke the model with the user's message
        response = model.invoke([HumanMessage(content=user_input)])
        
        print(f"AI: {response.content}\n")

if __name__ == "__main__":
    main()