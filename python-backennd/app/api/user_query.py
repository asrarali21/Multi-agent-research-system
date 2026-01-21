from fastapi import APIRouter
from pydantic import BaseModel





class UserQuery(BaseModel):
    query : str



router = APIRouter()






@router.post("/ask")
def user_query(query : UserQuery):

    return query