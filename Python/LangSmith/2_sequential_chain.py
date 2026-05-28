from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT']="Sequential Workflow"

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.7
)
llm2 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

model2=ChatHuggingFace(llm=llm2)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config={
    'run_name':'Sequential Chain',
    'tags':{'llm app','report generation','summarization'},
    'metadata':{'model1':'openai/gpt-oss-20b','model1_temp':0.7}
}

result = chain.invoke({'topic': 'Unemployment in India'},config=config)

print(result)
