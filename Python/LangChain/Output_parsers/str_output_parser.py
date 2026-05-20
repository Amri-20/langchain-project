from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, PromptTemplate

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='wite a 5 line summary on the given text {text}',
    input_variables=['text']
)

prompt1=template1.format(topic='black holes')
result1=model.invoke(prompt1)

prompt2=template2.format(text=result1.content)
result2=model.invoke(prompt2)

print(result2.content)