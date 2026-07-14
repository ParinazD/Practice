#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import langchain_core.messages as lcmessages



#---------------------------------------------
# Tool call example:
# NOTE: set the OPENROUTER_API_KEY in the terminal
#  example %~> export OPENROUTER_API_KEY=sdk_xyz
"""
============================================================
Math Assistant Chat
============================================================
Ask me math questions or anything else!
Available tools: addition, subtraction, multiplication
Type 'exit' or 'quit' to end the conversation

You: what is 5+5
  [Using tool: add with args: {'a': 5, 'b': 5}]

Assistant: 10

You: if I have 4 apples and I get 4 more what will be the total apples
  [Using tool: add with args: {'a': 4, 'b': 4}]

Assistant: You would have 8 apples in total.
"""
#---------------------------------------------


# -------------------------------------------------
# Define the Model
# -------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def get_openrouter_llm(max_tokens: int = 512, temperature: float = 0.0) -> ChatOpenAI:
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        max_tokens=max_tokens,
        temperature=temperature
    )


    # Bind the math tools to the LLM
    llm_with_tools = llm.bind_tools([add, subtract, multiply])
    return llm_with_tools




# Define math tools using the @tool decorator
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together. Returns the sum of a and b."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a. Returns the difference a - b."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Returns the product of a and b."""
    return a * b



def create_llm_with_tools(profile="brd3", region="us-east-1"):
    """Initialize and return the Bedrock LLM with math tools."""
    session = boto3.Session(profile_name=profile, region_name=region)
    bedrock_runtime = session.client('bedrock-runtime')

    llm = ChatBedrockConverse(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        client=bedrock_runtime,
        temperature=0.7,
        max_tokens=2000
    )

    # Bind the math tools to the LLM
    llm_with_tools = llm.bind_tools([add, subtract, multiply])
    return llm_with_tools



def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute the specified tool with the given arguments."""
    tools = {
        "add": add,
        "subtract": subtract,
        "multiply": multiply
    }

    if tool_name not in tools:
        return f"Error: Unknown tool '{tool_name}'"

    try:
        result = tools[tool_name].invoke(tool_args)
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"




def process_response(llm_with_tools, messages: list) -> str:
    """
    Process the LLM response and handle tool calls if needed.

    Args:
        llm_with_tools: The LLM instance with tools bound
        messages: The conversation message history

    Returns:
        The final response string
    """
    # Invoke the LLM
    ai_msg = llm_with_tools.invoke(messages)

    # Check if the model wants to use a tool
    if ai_msg.tool_calls:
        # Add the AI message to history
        messages.append(ai_msg)

        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"  [Using tool: {tool_name} with args: {tool_args}]")

            # Execute the tool
            tool_result = execute_tool(tool_name, tool_args)

            # Add tool result to messages
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_id))

        # Get the final response after tool execution
        final_response = llm_with_tools.invoke(messages)
        messages.append(final_response)

        return final_response.content
    else:
        # No tool calls, just return the direct response
        messages.append(ai_msg)
        return ai_msg.content



def chat_loop(llm_with_tools):
    """
    Run an interactive chat loop with the LLM and math tools.

    Args:
        llm_with_tools: The LLM instance with tools bound
    """
    # Initialize conversation history with system prompt
    system_prompt = (
        "You are a helpful math assistant. You have access to tools for addition, subtraction, and multiplication. "
        "Use these tools when the user asks for math operations. "
        "For non-math questions, provide helpful responses without using tools."
    )
    messages = [SystemMessage(content=system_prompt)]

    print("\n" + "="*60)
    print("Math Assistant Chat")
    print("="*60)
    print("Ask me math questions or anything else!")
    print("Available tools: addition, subtraction, multiplication")
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

            # Process the response (handles tool calls if needed)
            response = process_response(llm_with_tools, messages)

            # Display response
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nAssistant: Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.\n")


def main():
    """Main entry point."""
    # Create LLM instance with tools
    llm_with_tools = get_openrouter_llm() 

    # Start chat loop
    chat_loop(llm_with_tools)


if __name__ == '__main__':
    main()

