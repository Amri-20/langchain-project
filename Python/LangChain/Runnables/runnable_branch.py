from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnablePassthrough,RunnableSequence,RunnableParallel
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='write a detailed report on the topic {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='write a brief summary of the topic {text}',
    input_variables=['text']
)
parser=StrOutputParser()

report_chain=prompt1|model|parser

branch_chain=RunnableBranch(
    (lambda x:len(x.split())>500, RunnableSequence(prompt2,model,parser)),#lcel->prompt2|model|parser
    RunnablePassthrough()
)

final_chain=RunnableSequence(report_chain,branch_chain)

result=final_chain.invoke({'topic':'AI'})
print(result)
