from embedchain import App
from fastapi import APIRouter, responses
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(".env")

router = APIRouter()

# App config using OpenAI gpt-3.5-turbo-1106 as LLM
config = {
    # just a generic configration key
    "app": {
        "config": {
            "id": "chat-embeddings-app",
            "name":"chat-with-lecture"
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-3.5-turbo-1106",
        },
    },
}

# configuration to use Mistral as LLM
# app_config = {
#     "app": {
#         "config": {
#             "id": "embedchain-opensource-app"
#         }
#     },
#     "llm": {
#         "provider": "huggingface",
#         "config": {
#             "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
#             "temperature": 0.1,
#             "max_tokens": 250,
#             "top_p": 0.1
#         }
#     },
#     "embedder": {
#         "provider": "huggingface",
#         "config": {
#             "model": "sentence-transformers/all-mpnet-base-v2"
#         }
#     }
# }


ec_app = App.from_config(config=config)
# ec_app = App(config=config)


class SourceModel(BaseModel):
    note_id: str
    user: str 
    source: str


class QuestionModel(BaseModel):
    session_id: str
    question: str


@router.post("/api/v1/add")
async def add_source(source_model: SourceModel):
    """
    Adds a new source to the Embedchain app.
    Expects a JSON with a "source" key.
    """
    source = source_model.source

    ids = ec_app.db.get() # gets the metadata from the embedchain database and db is BaseVectorDB -> Configured vector database for the RAG app

    doc_hash = None
    # loop through the database to find the hash of our document based on note)id and user/user_id
    for meta_data in ids["metadatas"]:
        if (
            meta_data["note_id"] == source_model.note_id
            and meta_data["user"] == source_model.user
        ):
            doc_hash = meta_data["hash"]
            break

        # So if the document I am trying to add already exists , then Delete and re add
    if doc_hash:
        ec_app.delete(doc_hash)

    try:
        ec_app.add(
            source,
            metadata={"user": source_model.user, "note_id": source_model.note_id},
        )
        return {"message": f"Source '{source}' added successfully."}
    except Exception as e:
        response = f"An error occurred: Error message: {str(e)}."
        return {"message": response}


@router.get("/api/v1/search")
async def handle_search(query: str, user_id: str):
    """
    Handles a chat request to the Embedchain app.
    Accepts 'query' and 'session_id' as query parameters.
    """
    try:
        response = ec_app.query(query, citations=True, where={"user": {"$eq": user_id}}) # where parameter is used for metadata filtering -> A dictionary of key-value pairs to filter the chunks from the vector database
        # "$eq" is commonly used in database queries as an equality operator
    except Exception as e:
        response = f"An error occurred: Error message: {str(e)}"  # noqa:E501

    return response


@router.get("/")
async def root():
    print("hi")
    # return responses.RedirectResponse(url="/docs")
    return {"working"}

# if __name__ == "__main__":
#     pass