from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='Generate a linkdin post abut {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'linkedin':RunnableSequence(prompt1,model,parser),
    'tweet':RunnableSequence(prompt2,model,parser)
})
result=parallel_chain.invoke({'topic':'2 star on codechef'})
print(result.get('linkedin'))