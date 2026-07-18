import chromadb

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