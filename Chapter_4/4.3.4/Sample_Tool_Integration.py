# Sample_Tool_Integration.py
# Section 4.3.4
# Page 105

from langchain.tools import Tool

search_tool = Tool(
    name="Search",
    func=search_api_call,
    description="Useful for answering questions about current events or factual information."
)
