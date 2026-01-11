import os
from typing import Any

import requests
import weave

from kawai.tools.tool import KawaiTool, KawaiToolParameter


class WebSearchTool(KawaiTool):
    tool_name: str = "web_search"
    description: str = "Performs a google web search for your query then returns a string of the top search results."
    parameters: list[KawaiToolParameter] = [
        KawaiToolParameter(
            param_name="query",
            description="The search query to perform.",
            tool_type="string",
        ),
        KawaiToolParameter(
            param_name="filter_year",
            description="Optionally restrict results to a certain year",
            tool_type="number",
            required=False,
            nullable=True,
        ),
    ]

    def serper_api_results(self, query: str, filter_year: int | None = None):
        params = {
            "q": query,
            "api_key": os.getenv("SERPER_API_KEY"),
        }
        if filter_year is not None:
            params["tbs"] = (
                f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
            )
        response = requests.get("https://google.serper.dev/search", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(response.json())

    @weave.op
    def forward(self, query: str, filter_year: int | None = None) -> dict[str, Any]:
        results = self.serper_api_results(query=query, filter_year=filter_year)

        if "organic" not in results.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
        if "organic" not in results or len(results["organic"]) == 0:
            year_filter_message = (
                f" with filter year={filter_year}" if filter_year is not None else ""
            )
            return {
                "results": f"No results found for '{query}'{year_filter_message}."
                + " Try with a more general query, or remove the year filter."
            }

        snippets = []
        if "organic" in results:
            for idx, page in enumerate(results["organic"]):
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                snippets.append(redacted_version)

        return {"results": snippets}
