import os
from fastapi import APIRouter, HTTPException, responses
from pinecone import Pinecone, ServerlessSpec
import openai
from pydantic import BaseModel
from dotenv import load_dotenv
import itertools
import re
from openai import ChatCompletion

# Load environment variables
load_dotenv()

# Initialize Pinecone with API key
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "embeddings"

# Check if index exists, create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "text-embedding-ada-002"

# FastAPI router
router = APIRouter()

# Models for request data
class SourceModel(BaseModel):
    note_id: str
    user: str 
    source: str

class QuestionModel(BaseModel):
    session_id: str
    question: str

# Store session memory
session_memory = {}

# Generate embedding for text using OpenAI API
async def generate_embedding(text):
    response = openai.Embedding.create(input=text, model=MODEL_NAME)
    return response["data"][0]["embedding"]

@router.post("/api/v1/add")
async def add_source(source_model: SourceModel):
    try:
        source = source_model.source
        user = source_model.user
        note_id = source_model.note_id
        overlap_size = 20  # Adjust for desired overlap

        # Create overlapping chunks
        sentences = re.split(r'(?<=[.!?])\s+', source)
        chunks = [
            " ".join(sentences[i:i+3])  # Adjust number of sentences per chunk
            for i in range(0, len(sentences), 3 - overlap_size)
        ]

        # Embed and store each chunk
        for i, chunk in enumerate(chunks):
            embedding = await generate_embedding(chunk)
            chunk_metadata = {
                "user": user,
                "note_id": f"{note_id}-{i}",  # Unique ID for each chunk
                "text": chunk  # Store chunk text
            }
            
            # Upsert each chunk
            index.upsert(
                vectors=[{
                    "id": f"{note_id}-{user}-{i}",
                    "values": embedding,
                    "metadata": chunk_metadata
                }]
            )

        return {"message": "Source added successfully in overlapping chunks"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@router.put("/api/v1/update")
async def update_source(note_id: str, source_model: SourceModel):
    try:
        new_source = source_model.source
        user = source_model.user
        overlap_size = 20  # Adjust for desired overlap

        # Retrieve existing chunks for the given note_id
        existing_chunks = index.query(
            vector=[0] * 1536,  # Placeholder vector, since we only need metadata
            filter={"note_id": {"$regex": f"^{note_id}"}},  # Filter by note_id
            include_metadata=True,
            top_k=1000  # Adjust based on expected number of chunks
        )

        # Combine existing and new content
        existing_text = " ".join(
            [match["metadata"]["text"] for match in existing_chunks["matches"]]
        )
        combined_text = existing_text + " " + new_source

        # Create overlapping chunks from the combined text
        sentences = re.split(r'(?<=[.!?])\s+', combined_text)
        chunks = [
            " ".join(sentences[i:i+3])  # Adjust number of sentences per chunk
            for i in range(0, len(sentences), 3 - overlap_size)
        ]

        # Embed and store each chunk
        for i, chunk in enumerate(chunks):
            embedding = await generate_embedding(chunk)
            chunk_metadata = {
                "user": user,
                "note_id": f"{note_id}-{i}",  # Unique ID for each chunk
                "text": chunk  # Store chunk text
            }

            # Upsert each chunk
            index.upsert(
                vectors=[{
                    "id": f"{note_id}-{user}-{i}",
                    "values": embedding,
                    "metadata": chunk_metadata
                }]
            )

        return {"message": "Source updated successfully with new content"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


def get_relevant_context(query_embedding, top_k, user_id):
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"user": {"$eq": user_id}}
    )
    return [match["metadata"].get("text", "") for match in search_results["matches"]]

# The /chat endpoint with session-based memory
@router.post("/api/v1/chat")
async def chat(query: str, user_id: str):
    try:
        top_k = 3
        # Generate an embedding for the user query
        query_embedding = await generate_embedding(query)

        # Retrieve the top_k relevant contexts from the database
        context_list = get_relevant_context(query_embedding, top_k, user_id)
        print(context_list)
        # Create the custom prompt for the language model in a format compatible with gpt-3.5-turbo
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions based on provided context and the context is the lecture transcript that I attended."},
            {"role": "user", "content": f"Context: {context_list}\n\nQuestion: {query}\n\nPlease answer the question based only on the provided context."}
        ]

        # Call the language model to answer based on the context
        response = ChatCompletion.create(
            model="gpt-3.5-turbo",  # Updated model
            messages=messages,
            max_tokens=150,  # Adjust based on desired response length
            temperature=0.2  # Lower temperature to reduce creativity and "hallucination"
        )

        # Extract and return the model's response
        answer = response.choices[0].message["content"].strip()
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.get("/api/v1/search")
async def handle_search(query: str, user_id: str):
    try:
        query_embedding = await generate_embedding(query)

        # Retrieve the top relevant chunk
        search_results = index.query(
            vector=query_embedding,
            top_k=3,  # Top match
            include_metadata=True,
            filter={
                "user": {"$eq": user_id}  # Filter for the user
            },
        )

        if not search_results.get("matches"):
            return []

        # Retrieve only the top match's text and metadata
        response = [
            {
                "text": match["metadata"].get("text", ""),  # Relevant chunk
                "score": match.get("score", 0),
                "note_id": match["metadata"].get("note_id", "")
            }
            for match in search_results["matches"]
        ]
        
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )
    

# Root endpoint
@router.get("/")
async def root():
    return responses.RedirectResponse(url="/docs")
