import pinecone
from fastapi import APIRouter
import os

router = APIRouter()

# Initialize Pinecone client with environment variables
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # Your Pinecone API key
    environment=os.getenv("PINECONE_ENV")   # Pinecone environment
)

@router.get("/api/v1/admin/collections")
async def get_all_indexes():
    """
    Fetch all indexes (collections) from Pinecone.
    """
    indexes = pinecone.list_indexes()
    return {"indexes": indexes}

@router.get("/api/v1/admin/collections/{index_name}")
async def get_index_details(index_name: str):
    """
    Fetch details for a specific index in Pinecone.
    """
    try:
        index = pinecone.Index(index_name)
        stats = index.describe_index_stats()
        return {"index_name": index_name, "stats": stats}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Testing the API functions
    import asyncio

    async def test_get_all_indexes():
        return await get_all_indexes()

    all_indexes = asyncio.run(test_get_all_indexes())
    print(all_indexes)
