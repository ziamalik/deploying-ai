# Assignment 2 - Conversational AI System

This project contains a conversational AI agent designed as a Teaching Assistant for an AI course. 

## Features

### 1. Distinct Personality
The chatbot acts as a helpful and cheerful **Course Teaching Assistant**. It is programmed to politely decline questions on restricted topics (Cats, Dogs, Horoscopes, Zodiac Signs, Taylor Swift) and defends against system prompt injection.

### 2. Conversational Memory
The agent utilizes `langgraph.checkpoint.memory.MemorySaver` to persist conversational context across turns.

### 3. Integrated Services
The agent has three tool services at its disposal:
1. **API Calls (arXiv)**: Uses the public arXiv API to fetch recent academic research papers based on user queries. The response is transformed by an LLM (routed via the classroom API Gateway proxy to save costs) into an enthusiastic conversational description.
2. **Semantic Query (RAG)**: Uses native `chromadb.PersistentClient` to create a `course_materials` store. Embeddings are created via `OpenAIEmbeddingFunction` utilizing `text-embedding-3-small` going through the classroom AWS API Gateway to avoid arbitrary costs. The PDFs were parsed beforehand into a `dataset.csv` via `pdf_to_csv.py`.
3. **Web Search**: Integrates the `TavilySearchResults` tool to allow the agent to look up general queries or current news.

## Embedding Process

A separate script `pdf_to_csv.py` was used to process all PDF slides within `../01_materials/slides/`. The script uses `pypdf` to extract raw text content from each slide deck and writes it to a lightweight `dataset.csv` file (well under 40MB). 
During the first query to the teaching assistant, `CSVLoader` reads this file, embeds it using `text-embedding-3-small` through the OpenAI proxy Gateway, and saves it to a persistent local directory (`chroma_db/`) using native `chromadb`. A LangChain `BaseRetriever` class encapsulates the collection queries for seamless integration with the AI agent.

## Running the Application

1. Ensure the Python environment is activated and dependencies from `pyproject.toml` are installed.
2. Provide necessary API keys in `05_src/.secrets` (Required keys are `API_GATEWAY_KEY` and `TAVILY_API_KEY`).
3. Run the Gradio app:
```bash
cd 05_src/assignment_chat
python app.py
```
4. Access the conversational interface at `http://127.0.0.1:7860`.
