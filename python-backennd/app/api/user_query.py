from fastapi import APIRouter , HTTPException
from pydantic import BaseModel
from app.agents.coordinator_agent import CoordinatorAgent
from dotenv import load_dotenv 




load_dotenv()

class QueryRequest(BaseModel):
    query : str



router = APIRouter()

  
              
@router.post("/ask")
async def user_query(request : QueryRequest):


    coordinator = CoordinatorAgent()

    plan = await  coordinator.coordinate(request.query)

    return {
        "status" : "success",
        "coordination_plan":plan.model_dump()
    }