#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from langchain_openai import ChatOpenAI
import langchain_core.messages as lcmessages

#---------------------------------------------
# Most basic:
# Get a model object.
# invoke the model with System prompt + user prompt
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


systemPrompt = \
    """You are a helpful assistant that translates English to French.
        Translate the user sentence"""

messages = [lcmessages.SystemMessage(content=systemPrompt)]
messages.append(lcmessages.HumanMessage(content="I love programming"))


def main():
    llm = get_openrouter_llm()
    ai_msg = llm.invoke(messages)
    print("msg: ", ai_msg.content)  

if __name__ == '__main__':
    main()


