import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from os import environ as env
from fastapi.middleware.cors import CORSMiddleware
from api.routes import admin, api

load_dotenv()

app = FastAPI(title="Embedchain API")

app.include_router(api.router)
app.include_router(admin.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.middleware("http") #decorator is used to define a middleware function that will intercept incoming HTTP requests before they are processed by the main request handler and it will handle only http requests
# async def token_check_middleware(request: Request, call_next): #call_next is a callback function that will call the next middleware or the main request handler
#     token = request.headers.get("Authorization")

#     if request.url.path.startswith("/api/v1"):
#         if token != env.get("AUTH_TOKEN"):
#             raise HTTPException(status_code=401, detail="Unauthorized")
#     response = await call_next(request)
#     return response


# if __name__ == "__main__":
#     # print(pyth)
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         log_level="info",
#         reload=True,
#         timeout_keep_alive=600,
# )
