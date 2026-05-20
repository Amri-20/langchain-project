from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

loader = TextLoader('cricket.txt', encoding='utf-8')

docs=loader.load()

prompt=PromptTemplate(
    template='write a summary for the following poem -\n {poem}',
    input_variables=['poem']
)

parsers=StrOutputParser()

chain=prompt|model|parsers
result=chain.invoke({'poem':docs[0].page_content})
print(result)