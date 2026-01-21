from fastapi import FastAPI
from app.api.user_query import router as user_router

app = FastAPI(
    title="Multi Research Agent",
    version="1.0.0"
)



app.include_router(user_router)




@app.on_event("startup")
def startup_event():
    print("starting up")
