from firecrawl import Firecrawl
import os 


FIRE_CRAWL_API_KEY=os.getenv("FIRE_CRAWL_API_KEY")

firecrawl = Firecrawl(api_key=FIRE_CRAWL_API_KEY)


def firecrawl_scrape_tool(scrape_url:str):

    docs = firecrawl.scrape(scrape_url,formats=["markdown", "html"])

    return docs


def firecrawl_search_tool(query:str):

    result = firecrawl.search(
        query=query,
        limit=3
    )

    return result


def firecrawl_agent_tool(query:str):

    result = firecrawl.agent(
        query
    )

    return result

