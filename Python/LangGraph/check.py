import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env (Tumhara Hugging Face token yahi se uthega)
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# YAHAN HAI MAGIC: 
# Hum 'ChatOpenAI' use kar rahe hain (kyuki iska chat structure perfect hai)
# Par humne 'base_url' me Hugging Face ka pata daal diya hai! 
# Toh request OpenAI ke paas nahi, seedha Hugging Face ke paas jayegi aur bilkul FREE chalegi.
model = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct", # Tumhara pasandida model
    api_key=hf_token, 
    base_url="https://api-inference.huggingface.co/v1/", # HF Free Server
    max_tokens=256,
    temperature=0.7
)

# Prompt test kar
prompt = "What is LangGraph?"

print("Thinking...\n")
response = model.invoke(prompt)

# Output print karo (.content use karna padta hai chat models me)
print("AI Response:\n")
print(response.content)