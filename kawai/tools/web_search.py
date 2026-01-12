import os
from typing import Any

import requests
import weave

from kawai.tools.tool import KawaiTool, KawaiToolParameter


class WebSearchTool(KawaiTool):
    """Tool for performing Google web searches using the Serper API.

    This tool allows agents to search the web for current information. It uses
    the [Serper API](https://serper.dev) to perform Google searches and returns
    formatted results including titles, links, dates, sources, and snippets.

    Requires the `SERPER_API_KEY` environment variable to be set.

    Attributes:
        tool_name (str): Always "web_search"
        description (str): Brief description shown to the agent
        parameters (list[KawaiToolParameter]): Two parameters:
            - query (required, string): The search query
            - filter_year (optional, number): Restrict results to a specific year

    !!! example
        ```python
        from kawai import KawaiReactAgent, WebSearchTool

        agent = KawaiReactAgent(
            model="openai/gpt-4",
            tools=[WebSearchTool()],
            max_steps=5
        )

        # Agent will use this to search:
        # web_search(query="Python ReAct agents", filter_year=2024)
        ```

    Note:
        - Requires SERPER_API_KEY environment variable
        - Returns up to 10 search results by default (Serper API default)
        - Year filtering uses Google's date restriction syntax
        - If no results found, returns helpful error message
    """

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

    def serper_api_results(
        self, query: str, filter_year: int | None = None
    ) -> dict[str, Any]:
        """Call the Serper API to perform a Google search.

        Args:
            query (str): The search query string.
            filter_year (int | None): Optional year to restrict results (e.g., 2024).

        Returns:
            Raw JSON response from Serper API containing search results.

        Raises:
            ValueError: If the API request fails (non-200 status code).
        """
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
        """Execute a web search and return formatted results.

        Args:
            query (str): The search query to execute.
            filter_year (int | None): Optional year (e.g., 2024) to restrict results to that
                specific year.

        Returns:
            A dictionary with a "results" key containing either:
            - A list of formatted search result strings (if results found)
            - An error message string (if no results found)

            Each result string has the format:
            ```text
            0. [Title](https://example.com)
            Date published: YYYY-MM-DD
            Source: source_name
            snippet text...
            ```

        Raises:
            Exception: If no results found with year filter applied. This helps
                the agent understand it should retry without the filter.
            ValueError: If the Serper API request fails.

        !!! example
            ```python
            tool = WebSearchTool()
            results = tool.forward(query="Python asyncio", filter_year=2024)
            # Returns: {"results": ["0. [Title](https://example.com)\\n...", ...]}
            ```
        """
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
