from langchain_community.tools import DuckDuckGoSearchResults
import trafilatura

search = DuckDuckGoSearchResults(output_format="list")

def fetch_sites(query : str, top_k : int = 3):
    ret = search.invoke(query)
    fetched = []
    for row in ret[:top_k]:
        row["content"] = visit(row["link"])
        fetched.append(row)
    return fetched

def visit(link : str):
    content = trafilatura.fetch_url(link)
    return trafilatura.extract(content)
