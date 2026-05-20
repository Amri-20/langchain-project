from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel,RunnableBranch
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal['positive','negative']=Field(description="Give the sentiment of the feedback")


parser2=PydanticOutputParser(pydantic_object=feedback)

prompt1=PromptTemplate(
    template="Classify the sentiment of the following feedback: \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)


classifier_chain=prompt1|model|parser2

prompt2=PromptTemplate(
    template="Write an appropriate response to this positive feedback: \n {feedback}",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template="Write an appropriate response to this negative feedback: \n {feedback}",
    input_variables=["feedback"]
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment =='positive',prompt2|model|parser),
    (lambda x:x.sentiment =='negative',prompt3|model|parser),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain=classifier_chain|branch_chain

print(chain.invoke({"feedback":"I love the worst product ever!"}))

chain.get_graph().print_ascii()