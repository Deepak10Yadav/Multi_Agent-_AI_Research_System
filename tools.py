from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily = TavilyClient(api_key= os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query:str)->str:
    """Search the Web For Recent And Reliable Information On Our Topic. Return Titles, URLs And Snippets"""
    results = tavily.search(query=query,max_results=5)

    out = []
    for r in results['results']:
        out.append(
            f"Title:{r['title']}\n URL:{r['url']}\n Snippet:{r['content'][:300]}"
        )

    return "\n\n".join(out)


# print(web_search.invoke("What The Current Stock Market Updates"))

@tool
def scrap_url(url:str)->str:
    """Scrape and return clean text content from a given URL for deeper reading/research"""
    try:
        resp = requests.get(url,timeout=8,headers={"User-Agent":"Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","nav","footer"]):
            tag.decompose()
        return soup.get_text(separator=" ",strip=True)[:3000]
    except Exception as e:
        return f"Cloud Not Scrape URL:{str(e)}"

# print(scrap_url.invoke("URL OF Website"))