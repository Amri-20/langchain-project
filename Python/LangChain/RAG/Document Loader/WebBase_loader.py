from langchain_community.document_loaders import WebBaseLoader

# from langchain_community.document_loaders import TextLoader
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

url='https://www.amazon.in/s?bbn=81107433031&rh=n%3A81107433031%2Cp_85%3A10440599031&_encoding=UTF8&content-id=amzn1.sym.58c90a12-100b-4a2f-8e15-7c06f1abe2be&pd_rd_r=f66b4351-abda-4e1b-aa80-f6bc2304558d&pd_rd_w=qtcMr&pd_rd_wg=ziAtF&pf_rd_p=58c90a12-100b-4a2f-8e15-7c06f1abe2be&pf_rd_r=R2M0B12ATV1Y8WQKEH6H&ref=pd_hp_d_atf_unk'

loader=WebBaseLoader(url)

docs=loader.load()
prompt=PromptTemplate(
    template='Write a answer to the following question {question} based on following text {text}',
    input_variables=['question', 'text']
)

parsers=StrOutputParser()

chain=prompt|model|parsers

result=chain.invoke({'question':'which web page is this?','text':docs[0].page_content})

print(result)