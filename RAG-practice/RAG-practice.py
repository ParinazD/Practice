import chromadb

# Initialize the Persistent Client
# This automatically saves your database structure and data to the specified path
db_path = "./my_vector_db"

client = chromadb.PersistentClient(path=db_path)

# Create or get a collection
# Think of a collection like a table in a relational database
collection_name = "my_documents"
collection = client.get_or_create_collection(name=collection_name)

# 3. Add data to the collection
# By default, Chroma automatically handles generating embeddings using 'all-MiniLM-L6-v2'
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

print(f"Database successfully saved locally to: '{db_path}'")

# Verify it works by running a quick query
#querying the default text embedding model built into Chroma DB.
results = collection.query(
    query_texts=["Tell me about vector databases"],
    n_results=1
)

print("\nQuery Result:")
print(results["documents"])