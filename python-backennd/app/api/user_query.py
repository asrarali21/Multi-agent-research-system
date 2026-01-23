from fastapi import APIRouter
from pydantic import BaseModel
from app.agents.coordinator_agent import CoordinatorAgent
from dotenv import load_dotenv 




load_dotenv()

class UserQuery(BaseModel):
    query : str



router = APIRouter()

  
              
@router.post("/ask")
def user_query(query : UserQuery):


    coordination_service = CoordinatorAgent()


    response = coordination_service.coordinate(query)


    return response

