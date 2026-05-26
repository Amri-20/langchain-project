from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langgraph.graph.message import add_messages
load_dotenv()
from langgraph.checkpoint.memory import MemorySaver

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

checkpointer=MemorySaver()
graph=StateGraph(ChatState)

graph.add_node("Chat Node",chat_node)

graph.add_edge(START,"Chat Node")
graph.add_edge("Chat Node",END)

chatbot=graph.compile(checkpointer=checkpointer)

# for message_chunk,metadeta in chatbot.stream(
#     {'messages': [HumanMessage(content='What is the recipe to make poha')]},
#     config={"configurable": {"thread_id": "1"}},
#     stream_mode='messages'
#     ):

#     if message_chunk.content:
#         print(message_chunk.content,end=" ",flush=True)

# CONFIG = {"configurable": {"thread_id": "1"}}


# chatbot.invoke(
#                 {'messages': [HumanMessage(content='Hi my name is amrit')]},
#                 config=CONFIG,
#                 stream_mode='messages'
#             )

# print(chatbot.get_state(config=CONFIG).values['messages'])
