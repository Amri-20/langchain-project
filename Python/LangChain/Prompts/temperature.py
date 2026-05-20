from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm,temperature=0)
result=model.invoke("write a 5 line poem on crush")
model2=ChatHuggingFace(llm=llm,temperature=1.5)
result2=model2.invoke("write a 5 line poem on crush")
print("Temperature 0:\n",result.content)#same output for same input
print("\nTemperature 1.5:\n",result2.content)#different output for same input
print(result.content)