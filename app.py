from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
import gradio as gr
from langchain.agents import initialize_agent, AgentType

# === LLM ===
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# === RAG Агент ===
def run_mm_rag_agent(query: str) ->   str:
    from Multimodal_agent_RAG import agent_mm_rag  
    return agent_mm_rag.run(query)

def run_web_agent(query: str) ->   str:
    from real_time_market_agent import agent_executor 
    return agent_executor.invoke(query)

def run_analyse_agent(query: str) ->   str:# List[Document]:
    from DS_agent import dc_agent  
    return dc_agent.run(query)



mm_rag_tool = Tool(
    name="MultimodalRAG",
    func=run_mm_rag_agent,
    description=(
        "Useful when the user is asking for numerical, tabular, or visual document-related queries. "
        "This tool uses a multimodal retriever over structured documents including tables and charts."
    ),
)

web_rag_tool = Tool(
    name="WebsearchRAG",
    func=run_web_agent,
    description=(
        "Useful when the user requests data related to finding the most relevant information or information that is not contained in the financial statements of companies. "
        "This tool uses real-time search of information in network resources"
    ),
)
analyse_agent_tool= Tool(
    name="Analyse",
    func=run_analyse_agent,
    description=(
        "Useful when the user requests analysis and forecasting of time-varying data (monthly, quarterly, yearly, etc.) "
        "This tool forecasts the change in the analyzed value and visualizes this forecast"
    ),
)

supervisor_tools = [
    mm_rag_tool,
    web_rag_tool,
    analyse_agent_tool
    
]


research_agent = initialize_agent(
    tools=supervisor_tools,
    llm=llm,  # Или другой LLM
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=None, 
    #handle_parsing_errors=True # 
)

def chat_with_agent(user_input):
    try:
        response = research_agent.run(user_input)
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

iface = gr.Interface(
    fn=chat_with_agent,
    inputs=gr.Textbox(lines=4, placeholder="Enter your financial question..."),
    outputs="text",
    title="🧠 Financial RAG Super-Agent",
    description="Ask about financial metrics, forecasts, or recent market information."
)

if __name__ == "__main__":
    iface.launch()

# response = research_agent.run("What are Apple's net sales and long-term assets for the past 3 years?")
# print(response)

