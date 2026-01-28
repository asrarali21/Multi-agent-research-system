from firecrawl import Firecrawl

import os

FIRE_CRAWL_API_KEY=os.getenv("FIRE_CRAWL_API_KEY")

firecrawl =Firecrawl(
    api_key=FIRE_CRAWL_API_KEY
)