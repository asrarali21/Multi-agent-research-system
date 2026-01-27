from langchain_google_genai import ChatGoogleGenerativeAI

import os


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class ResearchPlan:

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

    def build_prompt(self , user_query:str)->str:

        return f"""
You will be given a research task by a user. Your job is to produce a set of instructions for a researcher that will complete the task. Do NOT complete the task yourself, just provide instructions on how to complete it.
GUIDELINES:
1. Maximize specificity and detail. Include all known user preferences and explicitly list key attributes or dimensions to consider.
2.If essential attributes are missing, explicitly state that they are open-ended.
3. Avoid unwarranted assumptions. Treat unspecified dimensions as flexible.
4. Use the first person (from the user's perspective).
5. When helpful, explicitly ask the researcher to include tables.
6. Include the expected output format (e.g. structured report with headers).
7.Preserve the input language unless the user explicitly asks otherwise.
8. Sources: prefer primary / official / original sources.]
{user_query}

"""
    def research(self , user_query):

        response = self.llm.ainvoke(self.build_prompt(user_query=user_query))

        return response