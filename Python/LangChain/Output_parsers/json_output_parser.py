from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template=PromptTemplate(
    template='give me 5 facts about {topic} /n {format_instruction}',
    input_variales=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt=template.format()

# result=model.invoke(prompt)

# final_result=parser.parse(result.content)

#---------OR-----------------

chain=template|model|parser

result=chain.invoke({'topic':'black holes'})
print(result)