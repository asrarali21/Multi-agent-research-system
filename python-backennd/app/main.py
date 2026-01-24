from dotenv import load_dotenv
import os 

from pathlib import Path


project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Verify it loaded
print(f"✅ .env loaded from: {env_path}")
print(f"🔑 GOOGLE_API_KEY exists: {bool(os.getenv('GOOGLE_API_KEY'))}")
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
