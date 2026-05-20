import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.set_page_config(page_title="Chatbot", layout="centered")

st.title("💬 Chatbot")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful AI assistant.")
    ]

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**AI:** {msg.content}")

# Input box (replacing dropdowns)
user_input = st.text_input("Type your message here...")

# Send button
if st.button("Send") and user_input:
    # Add user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get response
    result = model.invoke(st.session_state.chat_history)

    # Add AI response
    st.session_state.chat_history.append(AIMessage(content=result.content))

    # Rerun to update UI
    st.rerun()