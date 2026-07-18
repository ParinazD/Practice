import chromadb
import os 
import json
import requests

# Initialize the Persistent Client
# This automatically saves your database structure and data to the specified path
db_path = "./my_vector_db"

client = chromadb.PersistentClient(path=db_path)

# Create or get a collection
# Think of a collection like a table in a relational database
collection_name = "my_documents"
collection = client.get_or_create_collection(name=collection_name)

large_document = (
    "Artificial Intelligence has evolved rapidly over the last decade. "
    "A key breakthrough has been Retrieval-Augmented Generation, commonly known as RAG. "
    "RAG works by connecting a massive vector database to a generative language model. "
    "This architecture allows the AI to fetch factual context dynamically from private files, "
    "effectively eliminating hallucinations and ensuring up-to-date responses."
)

# Split the large document into smaller chunks (by sentence for this POC)
chunks = [chunk.strip() + "." for chunk in large_document.split(". ") if chunk]

# Add these precise text chunks to the vector database
print(f"Splitting text into {len(chunks)} chunks and embedding them...")
collection.add(
    documents=chunks,
    metadatas=[{"source": "ai_essay"} for _ in chunks],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# 3. Add data to the collection
# By default, Chroma automatically handles generating embeddings using 'all-MiniLM-L6-v2'
'''
collection.add(
    documents=[
        "Chroma DB is an open-source vector database used for AI applications.",
        "Python is a popular programming language for data science and machine learning.",
        "Vector databases store embeddings to allow for semantic similarity search."
    ],
    metadatas=[
        {"category": "tech"},
        {"category": "programming"},
        {"category": "tech"}
    ],
    ids=["doc1", "doc2", "doc3"]
)
'''

print(f"Database successfully saved locally to: '{db_path}'")

'''
# Verify it works by running a quick query
#querying the default text embedding model built into Chroma DB.

results = collection.query(
    query_texts=["Tell me about vector databases"],
    n_results=1
)
'''

# Query the database to see if it pulls the exact chunk instead of the whole page
user_question = "How does RAG eliminate hallucinations?"
print(f"\nUser Question: '{user_question}'")

#querying the default text embedding model built into Chroma DB.
search_results = collection.query(
    query_texts=[user_question],
    n_results=1
)
print("\nQuery Result:")
print(search_results["documents"])

#________________________________________________________________________________________________

# ---------- USING OPENROUTER API TO GENERATE ANSWERS BASED ON THE QUERY RESULTS ----------


from langchain.chat_models import ChatOpenAI

# Define the model you want to use (e.g., Google Gemini 2.5 Flash via OpenRouter)
MODEL_NAME = "openrouter/free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 

# Initialize the model with OpenRouter's base URL and config
llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model=MODEL_NAME,
    temperature=0.7
)

# Extract the retrieved text snippet from your Chroma search results
retrieved_context = search_results["documents"][0][0]


# Construct the prompt bundling the retrieved context and user question
rag_prompt = f"""
You are a helpful assistant. Answer the question using ONLY the provided context.

Context:
{retrieved_context}

Question:
{user_question}
"""

print(f"Sending prompt to OpenRouter ({MODEL_NAME})...")


# Make a POST request to OpenRouter's API endpoint for chat completions
try:
    #It sends the context chunk to OpenRouter,
    #waits for the response, parses the JSON data,
    #extracts the text answer, and prints it out.

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            # Optional headers required by OpenRouter for ranking
            "HTTP-Referer": "http://localhost:3000", 
            "X-Title": "RAG Practice App",
        },
        data=json.dumps({
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": rag_prompt}
            ]
        })
    )
    
    # Parse the response
    response_json = response.json()
    
    # STEP 1: Check if OpenRouter returned an error payload FIRST
    if "error" in response_json:
        print("--- OpenRouter API Error Payload Received ---")
        print(json.dumps(response_json["error"], indent=2))
        
    # STEP 2: Only look for choices if there is no error
    elif "choices" in response_json:
        llm_answer = response_json["choices"][0]["message"]["content"]
        print("--- GENERATED LLM RESPONSE ---")
        print(llm_answer)
        print("------------------------------")
        
    # STEP 3: Fallback for unexpected payloads
    else:
        print(" Unexpected response format from OpenRouter:")
        print(json.dumps(response_json, indent=2))

except Exception as e:
    print(f"\nAn error occurred while contacting OpenRouter: {e}")