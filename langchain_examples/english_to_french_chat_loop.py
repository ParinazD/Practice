#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


#---------------------------------------------
# Builds on the english_to_french_agent. Adds a chat loop. Keeps conversation history
# NOTE: set the OPENROUTER_API_KEY in the terminal
#  example %~> export OPENROUTER_API_KEY=sdk_xyz
#---------------------------------------------


# -------------------------------------------------
# Define the Model
# -------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def get_openrouter_llm(max_tokens: int = 512, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        max_tokens=max_tokens,
        temperature=temperature
    )




def chat_loop(llm, system_prompt):
    """
    Run an interactive chat loop with the LLM.
    
    Args:
        llm: The ChatBedrockConverse instance
        system_prompt: The system prompt to guide the assistant
    """
    # Initialize conversation history with system prompt
    messages = [SystemMessage(content=system_prompt)]
    
    print("\n" + "="*60)
    print("Chat with the Assistant")
    print("="*60)
    print("Type 'exit' or 'quit' to end the conversation\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nAssistant: Goodbye! Thanks for chatting.")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Add user message to history
            messages.append(HumanMessage(content=user_input))
            
            # Get response from LLM
            response = llm.invoke(messages)
            assistant_message = response.content
            
            # Add assistant message to history
            messages.append(AIMessage(content=assistant_message))
            
            # Display response
            print(f"\nAssistant: {assistant_message}\n")
            
        except KeyboardInterrupt:
            print("\n\nAssistant: Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.\n")





def main():
    """Main entry point."""
    # Create LLM instance
    llm = get_openrouter_llm() 

    # Define system prompt
    system_prompt = (
        "You are a helpful assistant that translates English to French. "
        "When the user provides English text, translate it to French. "
        "Be concise and accurate in your translations."
    )

    # Start chat loop
    chat_loop(llm, system_prompt)


if __name__ == '__main__':
    main()


