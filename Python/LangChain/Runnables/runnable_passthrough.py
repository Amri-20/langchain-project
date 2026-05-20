from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableSequence,RunnableParallel
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='give me a joke about the topic {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='give me a explanation about the topic {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)
result=final_chain.invoke({'topic':'AI'})

print(result)