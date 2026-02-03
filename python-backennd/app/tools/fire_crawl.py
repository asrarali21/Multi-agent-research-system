from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, List
import os
import httpx

FIRE_CRAWL_API_KEY = os.getenv("FIRE_CRAWL_API_KEY")


class FireCrawlSearchInput(BaseModel):
    """Input schema for FireCrawl search tool"""
    query: str = Field(description="The search query to find relevant web pages")
    num_results: int = Field(default=5, description="Number of search results to return (1-10)")


class FireCrawlSearchTool(BaseTool):
    """
    FireCrawl web search tool that scrapes and extracts content from web pages.
    """
    name: str = "firecrawl_web_search"
    description: str = """
    Useful for searching the web and getting detailed content from web pages.
    Use this when you need:
    - Current information from the internet
    - News articles, blog posts, documentation
    - Statistics, data, research findings from web sources
    
    Input: A search query string
    Output: List of web pages with titles, URLs, and extracted content
    """
    args_schema: Type[BaseModel] = FireCrawlSearchInput
    
    api_key: str = FIRE_CRAWL_API_KEY
    base_url: str = "https://api.firecrawl.dev/v1"
    
    def _run(self, query: str, num_results: int = 5) -> str:
        """Synchronous version (not used, but required by BaseTool)"""
        raise NotImplementedError("Use async version with _arun")
    
    async def _arun(self, query: str, num_results: int = 5) -> str:
        """
        Async execution of FireCrawl search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Formatted string with search results
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Search using FireCrawl
                search_response = await client.post(
                    f"{self.base_url}/search",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "query": query,
                        "limit": num_results,
                        "scrapeOptions": {
                            "formats": ["markdown", "html"],
                            "onlyMainContent": True
                        }
                    }
                )
                
                if search_response.status_code != 200:
                    return f"Error searching: {search_response.text}"
                
                results = search_response.json()
                
                # Step 2: Format results
                if not results.get("data"):
                    return "No results found for this query."
                
                formatted_results = []
                for idx, result in enumerate(results["data"][:num_results], 1):
                    formatted_results.append(
                        f"## Result {idx}: {result.get('title', 'Untitled')}\n"
                        f"**URL**: {result.get('url', 'N/A')}\n"
                        f"**Content**:\n{result.get('markdown', result.get('content', 'No content available'))}\n"
                        f"---\n"
                    )
                
                return "\n".join(formatted_results)
                
        except Exception as e:
            return f"Error using FireCrawl: {str(e)}"


class FireCrawlScrapeTool(BaseTool):
    """
    FireCrawl tool to scrape a specific URL and extract its content.
    """
    name: str = "firecrawl_scrape_url"
    description: str = """
    Scrapes content from a specific URL using FireCrawl.
    Use this when you have a specific URL you want to extract content from.
    
    Input: A URL string
    Output: Extracted markdown content from the URL
    """
    
    api_key: str = FIRE_CRAWL_API_KEY
    base_url: str = "https://api.firecrawl.dev/v1"
    
    def _run(self, url: str) -> str:
        """Synchronous version (not used)"""
        raise NotImplementedError("Use async version with _arun")
    
    async def _arun(self, url: str) -> str:
        """
        Async scraping of a specific URL.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/scrape",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "url": url,
                        "formats": ["markdown"],
                        "onlyMainContent": True
                    }
                )
                
                if response.status_code != 200:
                    return f"Error scraping URL: {response.text}"
                
                result = response.json()
                return result.get("data", {}).get("markdown", "No content extracted")
                
        except Exception as e:
            return f"Error scraping URL: {str(e)}"