import os
from dotenv import load_dotenv

# Load environment variables before any tool initialization
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.secrets'))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from services import get_arxiv_summary, get_rag_retriever, get_web_search_tool
from guardrails import get_system_prompt

# Define the tools
@tool
def arxiv_search_tool(query: str) -> str:
    """Useful for searching and summarizing recent academic research papers on arXiv. The user should provide a topic or query."""
    return get_arxiv_summary(query)

@tool
def course_materials_search(query: str) -> str:
    """Useful for searching the course materials (PDF slides) for answers to questions about AI, LLMs, RAG, Prompt Engineering, etc."""
    retriever = get_rag_retriever()
    docs = retriever.invoke(query)
    # Combine the top results
    if docs:
        result = "\n\n".join([f"Source: {doc.metadata.get('source_file')}\nContent: {doc.page_content}" for doc in docs])
        return result
    return "No relevant course materials found for this query."

def get_agent():
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
        api_key="any value",
        default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY", "")}
    )

    # Initialize web search tool here (after env vars are loaded)
    web_search_tool = get_web_search_tool()

    # Define our tools list
    tools_list = [arxiv_search_tool, course_materials_search, web_search_tool]
    
    # Create the graph agent with memory
    memory = MemorySaver()
    
    agent_executor = create_react_agent(
        llm, 
        tools_list, 
        prompt=get_system_prompt(),
        checkpointer=memory
    )
    
    return agent_executor
