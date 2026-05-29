from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

print(model.invoke("What is your name"))