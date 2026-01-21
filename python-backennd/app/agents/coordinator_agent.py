from langchain_google_genai import ChatGoogleGenerativeAI
import os
GOOGLE_APIKEY=os.getenv("GOOGLE_APIKEY")

class CoordinatorAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash",
            api_key=GOOGLE_APIKEY
        )

    def build_prompt(self):

        prompt ="""
You are coordination agent where you determine the user query 


"""
