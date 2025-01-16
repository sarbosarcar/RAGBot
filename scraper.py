from langchain_community.tools import DuckDuckGoSearchResults
import trafilatura

import re

search = DuckDuckGoSearchResults(output_format="list")

def parse_results(results_str):
    # Define a regular expression pattern to extract each entry
    pattern = r"snippet: (.*?), title: (.*?), link: (https?://[^\s]+)"
    matches = re.findall(pattern, results_str)
    
    # Convert matches to a list of dictionaries
    parsed_results = [
        {"snippet": snippet.strip(), "title": title.strip(), "link": link.strip()}
        for snippet, title, link in matches
    ]
    return parsed_results

def fetch_sites(query : str, top_k : int = 3):
    ret = search.invoke(query)
    #ret = parse_results(ret)
    fetched = []
    for row in ret[:top_k]:
        row["content"] = visit(row["link"])
        fetched.append(row)
    return fetched

def visit(link : str):
    content = trafilatura.fetch_url(link)
    return trafilatura.extract(content)
