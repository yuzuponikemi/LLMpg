# search_tool.py
from duckduckgo_search import DDGS
import requests
import json
from google.generativeai.types import FunctionDeclaration, Tool

# Import the logger
from src.logger.agent_logger import setup_logger

# Setup logger for this module
logger = setup_logger(logger_name="search_tool")

def search_duckduckgo(query: str, max_results: int = 5) -> str:
    """
    Performs a web search using DuckDuckGo and returns the results.

    Args:
        query: The search query string.
        max_results: The maximum number of search results to return.

    Returns:
        A string containing the formatted search results,
        or an error message if the search fails.
    """
    logger.info(f"Calling Search Tool with query: {query}")
    try:
        # DDGS().text returns dictionaries for each result
        results = DDGS().text(query, max_results=max_results)
        logger.debug(f"Search results: {results}")
        if not results:
            logger.warning("No search results found.")
            return "No results found."

        # Format the results into a single string
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append(
                f"Result {i+1}:\nTitle: {result.get('title', 'N/A')}\n"
                f"Snippet: {result.get('body', 'N/A')}\n"
                f"URL: {result.get('href', 'N/A')}\n"
            )
        logger.info(f"Returning {len(formatted_results)} formatted search results")
        return "\n---\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        return f"An error occurred during the search: {e}"

# --- Tool Schema Definition (for Gemini) ---
search_duckduckgo_declaration = FunctionDeclaration(
    name="search_duckduckgo",
    description="Searches the web using DuckDuckGo for a given query and returns the top results. Use this for general web searches to find information or relevant URLs.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query."
            }
        },
        "required": ["query"]
    }
)

# You might want a Tool object if your setup uses it directly
search_tool_gemini = Tool(
    function_declarations=[search_duckduckgo_declaration],
)

# Example Usage (optional - you can run this file directly to test)
if __name__ == "__main__":
    logger.info("Running search_tool.py as main script")
    test_query = "What is the latest news on Thinkcyte?"
    search_results = search_duckduckgo(test_query)
    print("\n--- Search Results ---")
    print(search_results)
    print("----------------------")
    logger.info("Test search completed")
