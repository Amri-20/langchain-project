import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import BaseMessage,HumanMessage
import uuid

# ************************************UTILITY ID***************************
def generate_thread_id():
    thread_id=uuid.uuid4()
    return thread_id

def new_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    st.session_state['message_history']=[]
    add_thread(st.session_state['thread_id'])

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def resume_chat(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}
).values['messages']

# ************************************SESSION SETUP************************


if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]
# message_history=[] not used cause it will always be empty at each input


if'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=[]

add_thread(st.session_state['thread_id'])



# ************************************SIDEBAR******************************


st.sidebar.title('LangGraph ChatBot')

if st.sidebar.button('New Chat'):
    new_chat()

st.sidebar.header("My Conv.")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id']=thread_id
        messages=resume_chat(thread_id)

        temp_msg=[]
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_msg.append({'role':role,'content':msg.content})
        
        st.session_state['message_history']=temp_msg




# ************************************UI***********************************
# Loading back messages
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
    

user_input=st.chat_input("Type Here")

if user_input:
    #first add the history to the message_history
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    # CONFIG={'configurable': st.session_state['thread_id']}
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}


    with st.chat_message('assistant'):
        ai_output=st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role':'assistant','content':ai_output})