from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Annotated
from dotenv import load_dotenv
import requests

# load env variables
load_dotenv()


# =========================
# TOOL 1
# =========================
@tool
def get_conversion_factor(
    base_currency: str,
    target_currency: str
) -> dict:
    """
    Fetch real-time currency conversion rate
    """

    url = f"https://v6.exchangerate-api.com/v6/e95ce53494bf36351a73e88c/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    return response.json()


# =========================
# TOOL 2
# =========================
@tool
def converter(
    base_currency_value: int,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """
    Convert currency using conversion rate
    """

    return base_currency_value * conversion_rate


# =========================
# MODEL
# =========================
llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

model_with_tools = model.bind_tools(
    [get_conversion_factor, converter]
)


# =========================
# USER MESSAGE
# =========================
messages = [
    HumanMessage(
        content="What is the conversion factor between USD and INR and convert 10 USD to INR"
    )
]


# =========================
# FIRST MODEL CALL
# =========================
ai_msg = model_with_tools.invoke(messages)

print("\nAI TOOL CALLS:")
print(ai_msg.tool_calls)

messages.append(ai_msg)


# =========================
# EXECUTE TOOLS
# =========================
for tool_call in ai_msg.tool_calls:

    # -------------------------
    # TOOL 1
    # -------------------------
    if tool_call["name"] == "get_conversion_factor":

        # execute tool
        result = get_conversion_factor.invoke(
            tool_call["args"]
        )

        print("\nTOOL 1 RESULT:")
        print(result)

        # fetch conversion rate
        conversion_rate = result["conversion_rate"]

        # append tool message
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

        # -------------------------
        # TOOL 2
        # -------------------------
        converted_amount = converter.invoke({
            "base_currency_value": 10,
            "conversion_rate": conversion_rate
        })

        print("\nFINAL CONVERTED VALUE:")
        print(converted_amount)


# =========================
# FINAL OUTPUT
# =========================
print(f"\n10 USD = {converted_amount} INR")