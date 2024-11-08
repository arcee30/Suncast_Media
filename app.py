import os
import openai
import numpy as np
import streamlit as st
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from tqdm import tqdm

# Load API key from environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load and vectorize the documents
def get_embedding(text):
    """Generate an embedding for a given text."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response['data'][0]['embedding'])

def load_and_embed_documents(folder_path="transcripts"):
    embeddings = []
    documents = []
    
    # Load each .txt file, store the text and its embedding
    for filename in tqdm(os.listdir(folder_path), desc="Embedding documents"):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text = file.read()
                documents.append(text)
                embeddings.append(get_embedding(text))
                
    return documents, embeddings

# Load and embed documents once
documents, embeddings = load_and_embed_documents("transcripts")

# Step 2: Streamlit app for query input and retrieval
st.title("RAG Assistant")

query = st.text_input("Ask your question:")
if query:
    query_embedding = get_embedding(query)
    
    # Find the document with the smallest cosine distance to the query
    distances = [cosine(query_embedding, doc_embedding) for doc_embedding in embeddings]
    best_match_index = np.argmin(distances)
    
    # Retrieve the best-matching document text
    best_document = documents[best_match_index]
    
    # Use the retrieved document to craft a response with OpenAI
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Given the context:\n\n{best_document}\n\nAnswer the following question:\n{query}",
        max_tokens=150
    )
    
    # Display the response
    st.write(response.choices[0].text.strip())
