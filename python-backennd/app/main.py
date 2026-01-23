from fastapi import FastAPI
from app.api.user_query import router as user_router
from dotenv import load_dotenv
import os 



load_dotenv()
print(f"🔑 GOOGLE_API_KEY exists: {bool(os.getenv('GOOGLE_API_KEY'))}")
print(f"🔑 API Key first 10 chars: {os.getenv('GOOGLE_API_KEY', '')[:10]}...")
app = FastAPI(
    title="Multi Research Agent",
    version="1.0.0"
)



app.include_router(user_router)




@app.on_event("startup")
def startup_event():
    print("starting up")
