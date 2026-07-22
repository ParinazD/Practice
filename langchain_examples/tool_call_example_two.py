#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import langchain_openai
import langchain.tools as lctools
import langchain.messages as lcmessages

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")



# Define tools
@lctools.tool
def get_weather_information(city: str):
    """Return a JSON structure with weather information for the city.
       Arguments:
       city: str - Name of the city
    """
    windSpeed = random.randint(2, 300)
    airQuality = random.choice(["clean", "smog", "smoke", "toxic"])
    humidity = random.randint(2, 60)
    temperature = random.randint(-15, 120)

    return {
        "temperature": temperature,
        "temperature_unit": "Firenhiet",
        "wind_speed": windSpeed,
        "humidity": humidity,
        "air_quality": airQuality,
        "sunrise": "6:30 AM",
        "sunset": "7:15 PM",
    }


# Define a model



def get_openrouter_model(openrouterModelId: str= "openai/gpt-oss-20b",
                         openrouterBaseUrl="https://openrouter.ai/api/v1",
                         temperature: float="0.7"):
    llm = langchain_openai.ChatOpenAI(
        model=openrouterModelId,
        openai_api_base=openrouterBaseUrl,
        openai_api_key=OPENROUTER_API_KEY,
        max_tokens=512,
        temperature=temperature
    )
    llmWithTools = llm.bind_tools([get_weather_information])
    return llmWithTools


def agent_loop(llm, messages):
    """
    The Agent Loop of "Reasoning-Acting" REACT
    """
    while True:
        response = llm.invoke(messages)
        messages.append(response)

        finishReason = response.response_metadata.get("finish_reason")
        print("Finish Reason: ", finishReason)

        if finishReason !=  "tool_calls":
            return response.content

        for call in response.tool_calls:
            toolName = call["name"]
            toolArgs = call["args"]
            toolId = call["id"]

            print(f"Tool name: {toolName}, args: {toolArgs}, id: {toolId}")

            # Execute the tool
            # Since we only have one we will simplify
            if toolName == "get_weather_information":
                toolResult = get_weather_information.invoke(toolArgs)
                messages.append(lcmessages.ToolMessage(content=str(toolResult),
                                                       tool_call_id=toolId))


def chat_loop(llm):
    """Chat loop"""
    systemPrompt = \
        """
        You are a helpful assistant with a sense of humor, to help user answer questions related to weather
        for various locations that user asks for.
        Instructions:
        1. Use the tools available to get weather information
        2. If conditions are not suitable, you will recommend to not visit the location.
        Output:
        Your output should be consise and accurate based on the tool responses, along
        with a recommendation of whether its a good time to visit.
        Add something funny, for example if the wind spped is extreme, you can add
        "May I recommend Mars. The conditions seem more suitable for you there :) ."
        """

    messages = [lcmessages.SystemMessage(content=systemPrompt)]

    while True:
        userInput = input("You: ").strip()

        if userInput.lower() in ["q", "quit", "exit"]:
            break

        if not userInput:
            print("Agent: ...")
            continue

        messages.append(lcmessages.HumanMessage(content=userInput))

        response = agent_loop(llm, messages)
        print("Agent: ", response)


def main():
    llm = get_openrouter_model()

    # Start the chat loop
    chat_loop(llm)


if __name__ == '__main__':
    main()
