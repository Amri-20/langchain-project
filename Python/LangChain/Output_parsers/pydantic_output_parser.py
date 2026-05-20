from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

# ✅ Define schema using Pydantic
class Facts(BaseModel):
    fact_1: str = Field(description="Fact 1 about the topic")
    fact_2: str = Field(description="Fact 2 about the topic")
    fact_3: str = Field(description="Fact 3 about the topic")

# ✅ Parser
parser = PydanticOutputParser(pydantic_object=Facts)

# ✅ Model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# ✅ Prompt
template = PromptTemplate(
    template="""
Give 3 facts about {topic}.

{format_instructions}

STRICTLY return ONLY valid JSON.
""",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# ✅ Chain
chain = template | model | parser

# ✅ Run
result = chain.invoke({"topic": "black hole"})
print(result)