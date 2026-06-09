from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

model = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

class RouterDecision(BaseModel):
    action: str
    reason: str

decider = model.with_structured_output(RouterDecision)

result = decider.invoke("Decide what to do with this query: Write a blog on AI")

print(result)