import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import BaseMessage,HumanMessage

if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]
# message_history=[] not used cause it will always be empty at each input

# Loading back messages
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
    
# {'role':'user','content':'Hi'}
# {'role':'assistant','content':'Hi,how can i assist you'}


user_input=st.chat_input("Type Here")


if user_input:
    #first add the history to the message_history
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    with st.chat_message('assistant'):
        ai_output=st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": "1"}},
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role':'assistant','content':ai_output})