from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langgraph.graph.message import add_messages
load_dotenv()
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

def chat_node(state:ChatState)->ChatState:
    messages=state['messages']

    response=model.invoke(messages)

    return{
        'messages':[response]
    }


conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)


checkpointer=SqliteSaver(conn=conn)


graph=StateGraph(ChatState)

graph.add_node("Chat Node",chat_node)

graph.add_edge(START,"Chat Node")
graph.add_edge("Chat Node",END)

chatbot=graph.compile(checkpointer=checkpointer)

# Taking all checkpoint value
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