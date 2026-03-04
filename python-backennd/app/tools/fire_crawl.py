from firecrawl import FirecrawlApp
from langchain_core.tools import tool
import os
import json

FIRE_CRAWL_API_KEY = os.getenv("FIRE_CRAWL_API_KEY")

firecrawl = FirecrawlApp(api_key=FIRE_CRAWL_API_KEY)


@tool
def firecrawl_search(query: str) -> str:
    """Search the web for information on a given query. Returns relevant search results with URLs and content snippets. Use this to find sources and information about a topic."""
    try:
        result = firecrawl.search(query=query, limit=5)
        if isinstance(result, dict) and "data" in result:
            formatted = []
            for item in result["data"][:5]:
                title = item.get("title", "No title")
                url = item.get("url", "")
                snippet = item.get("markdown", item.get("description", ""))[:500]
                formatted.append(f"**{title}**\nURL: {url}\n{snippet}\n")
            return "\n---\n".join(formatted) if formatted else "No results found."
        return json.dumps(result, default=str)[:3000]
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def firecrawl_scrape(url: str) -> str:
    """Scrape a specific URL and return its content as markdown. Use this when you have a URL you want to read in detail."""
    try:
        result = firecrawl.scrape_url(url, formats=["markdown"])
        if isinstance(result, dict):
            content = result.get("markdown", result.get("content", ""))
            if content:
                return content[:5000]  
            return json.dumps(result, default=str)[:3000]
        return str(result)[:3000]
    except Exception as e:
        return f"Scrape error: {str(e)}"



research_tools = [firecrawl_search, firecrawl_scrape]
