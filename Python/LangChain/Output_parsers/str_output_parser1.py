from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

parser=StrOutputParser()
chain=template1 |model| parser| template2| model|parser

result=chain.invoke({'topic':'black holes'})

print(result)