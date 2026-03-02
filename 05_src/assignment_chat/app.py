import gradio as gr
from agent import get_agent
from guardrails import check_input_guardrails
from langchain_core.messages import HumanMessage


# Initialize the agent
agent = get_agent()

# Define a thread id to track the conversation in memory
config = {"configurable": {"thread_id": "chat_thread"}}

def chat_response(message, history):
    # 1. Guardrail Check
    guardrail_error = check_input_guardrails(message)
    if guardrail_error:
        return guardrail_error
    
    # 2. Agent invocation
    try:
        inputs = {"messages": [HumanMessage(content=message)]}
        responses = agent.invoke(inputs, config=config)
        
        # The last message is the AI's final response
        return responses["messages"][-1].content
    except Exception as e:
        return f"Whoops! I encountered an error while trying to help you: {e}"

# Build the Gradio interface
chat = gr.ChatInterface(
    fn=chat_response,
    type="messages",
    title="AI Course Teaching Assistant",
    description="Hello! I am your AI Course TA. Feel free to ask me questions about the course materials, recent arXiv research papers maintained by Cornell University, or any general topic via web search!",
    examples=[
        "Can you summarize recent research papers on multi-agent LLM systems?",
        "Explain Prompt Engineering based on the course materials.",
        "Search the web for the latest updates on Large Language Models.",
        "Find recent arXiv papers on Retrieval Augmented Generation.",
        "What are foundation models according to the course slides?",
        "What is the current state of AI regulation worldwide?",
        "How does our course explain RAG, and are there recent arXiv papers that expand on it?",
        "Compare what our course says about LLMs with recent arXiv research and current industry news from the web.",
    ]
)

if __name__ == "__main__":
    chat.launch(server_name="127.0.0.1", server_port=7860)
