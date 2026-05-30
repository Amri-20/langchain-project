from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import sqlite3
import requests
import random


load_dotenv()

# -------------------
# 1. LLM
# -------------------

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

loader=PyPDFLoader("intro-to-ml.pdf")
docs=loader.load()

splitters=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitters.split_documents(docs)

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore=FAISS.from_documents(chunks,embedding)

retriever=vectorstore.as_retriever(search_type='similarity',search_kwargs={'k':4})

# -------------------
# 2. Tools
# -------------------
# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(query):
    """
    Retrieve relevant information from the pdf document.
    """

    result = retriever.invoke(query)

    context = [docs.page_content for docs in result]
    metadata = [docs.metadata for docs in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata
    }

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=DJUYBPF2QW1526V6"
    r = requests.get(url)
    return r.json()


tools = [search_tool, get_stock_price, calculator]
model_with_tools = model.bind_tools(tools)


# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def chat_node(state:ChatState)->ChatState:
    messages=state['messages']

    response=model_with_tools.invoke(messages)

    return{
        'messages':[response]
    }

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)


checkpointer=SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
# graph=StateGraph(ChatState)

# graph.add_node("Chat Node",chat_node)

# graph.add_edge(START,"Chat Node")
# graph.add_edge("Chat Node",END)


# graph structure
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)   

chatbot=graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def thread_history():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
         all_threads.add(checkpoint.config['configurable']['thread_id'])
    
    return list(all_threads)



# CONFIG = {"configurable": {"thread_id": "thread-1"}}


# response=chatbot.invoke(
#                 {'messages': [HumanMessage(content='What is my name?')]},
#                 config=CONFIG,
#                 # stream_mode='messages'
#             )

# print(response)

result=model.invoke("what is the token limit of huggingface free api")
print(result.content)