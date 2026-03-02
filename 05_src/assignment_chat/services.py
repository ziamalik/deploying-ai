import os
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma

import xml.etree.ElementTree as ET

# Service 1: API Call (arXiv)
def fetch_arxiv_papers(query: str, max_results: int = 2):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title_el = entry.find('{http://www.w3.org/2005/Atom}title')
            summary_el = entry.find('{http://www.w3.org/2005/Atom}summary')
            
            title = title_el.text.strip() if title_el is not None and title_el.text else "No Title"
            summary = summary_el.text.strip() if summary_el is not None and summary_el.text else "No Summary"
            
            # Clean up newlines in summary
            summary = " ".join(summary.split())
            papers.append({"title": title, "summary": summary})
        return papers
    except Exception as e:
        return {"error": str(e)}

def get_arxiv_summary(query: str) -> str:
    """Fetches recent arXiv papers and transforms the summaries to a natural tone using an LLM."""
    papers = fetch_arxiv_papers(query)
    
    if isinstance(papers, dict) and "error" in papers:
        return f"Sorry, I couldn't fetch research papers right now: {papers['error']}"
        
    if not isinstance(papers, list) or not papers:
        return f"I couldn't find any recent papers on arXiv for '{query}'."
    
    # Limit to first 2 papers to avoid context overflow
    formatted_papers = ""
    for idx, paper in enumerate(papers[:2]):
        formatted_papers += f"\n\nPaper {idx+1} Title: {paper['title']}\nAbstract: {paper['summary']}"
        
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.7,
        base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
        api_key="any value",
        default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY", "")}
    )
    prompt = f"You are a helpful AI Course Teaching Assistant. I just ran a search for recent academic papers on '{query}'. Here are the raw results:\n{formatted_papers}\n\nPlease summarize these findings into a concise, enthusiastic, and highly accessible conversational format for a student. Do not reveal raw formatting or talk like a robot."
    
    response = llm.invoke(prompt)
    return response.content

# Service 2: RAG
def get_rag_retriever():
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    csv_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    embedding_fn = OpenAIEmbeddingFunction(
        api_key="any value",
        model_name="text-embedding-3-small",
        api_base="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
        default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY", "")}
    )
    
    collection = chroma_client.get_or_create_collection(
        name="course_materials",
        embedding_function=embedding_fn
    )
    
    if collection.count() == 0 and os.path.exists(csv_path):
        loader = CSVLoader(file_path=csv_path, source_column="source_file", encoding="utf-8")
        docs = loader.load()
        
        documents = [doc.page_content for doc in docs]
        metadatas = [{"source_file": doc.metadata.get("source_file", "unknown")} for doc in docs]
        ids = [f"id{i}" for i in range(len(docs))]
        
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
    
    # Wrap the chroma collection in a BaseRetriever for LangChain compatibility
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from typing import List
    from pydantic import PrivateAttr
    
    class ChromaCollectionRetriever(BaseRetriever):
        _collection: chromadb.Collection = PrivateAttr()
        
        def __init__(self, collection: chromadb.Collection, **kwargs):
            super().__init__(**kwargs)
            self._collection = collection
            
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            results = self._collection.query(
                query_texts=[query],
                n_results=4
            )
            docs = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    docs.append(Document(
                        page_content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                    ))
            return docs

    return ChromaCollectionRetriever(collection=collection)
    
# Service 3: Web Search
def get_web_search_tool():
    """Returns a tool capable of performing web searches via Tavily."""
    return TavilySearchResults(max_results=3)
