import operator
from typing import TypedDict, List, Annotated, Literal
from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import RetryWithErrorOutputParser

load_dotenv()

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(description="One sentence describing what the reader should be able to do/understand after this section.")
    # FIXED: min_length=0 to prevent crash if model returns empty list []
    bullets: List[str] = Field(min_length=0, max_length=5, description="Up to 5 concrete, non-overlapping subpoints.")
    target_words: int = Field(description="Target word count for this section (120-450).")
    section_type: Literal["intro", "core", "examples", "checklist", "common_mistakes", "conclusion"]

class Plan(BaseModel):
    blog_title: str
    audience: str = Field(description="Who this blog is for.")
    tone: str = Field(description="Writing tone (e.g., practical, crisp).")
    tasks: list[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    result: str

# FIXED: Using a model capable of strict JSON generation and increased max_new_tokens
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=2500,
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)
parser = PydanticOutputParser(pydantic_object=Plan)
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)

def orchestrator(state: State) -> dict:
    format_instructions = parser.get_format_instructions()
    
    # FIXED: ChatPromptValue wrapper prevents 'to_string' error in retry_parser
    prompt_value = ChatPromptValue(messages=[
        SystemMessage(
            content=f"""You are a senior technical writer and developer advocate. Produce a highly actionable outline for a technical blog post.

CRITICAL SCHEMA RULES:
1. EVERY task MUST have up to 5 items in the `bullets` list.
2. EVERY task MUST include ALL 6 fields: `id`, `title`, `goal`, `bullets`, `target_words`, and `section_type`.

YOU MUST RETURN YOUR ANSWER AS A VALID JSON OBJECT EXACTLY MATCHING THIS TEMPLATE:
{{
  "blog_title": "Example Title",
  "audience": "Intermediate developers",
  "tone": "Practical",
  "tasks": [
    {{
      "id": 1,
      "title": "Introduction",
      "goal": "Understand the core problem.",
      "bullets": [
        "Point 1",
        "Point 2"
      ],
      "target_words": 200,
      "section_type": "intro"
    }},
    {{
      "id": 2,
      "title": "Core Concepts",
      "goal": "Learn the architecture.",
      "bullets": [
        "Point A",
        "Point B"
      ],
      "target_words": 300,
      "section_type": "core"
    }}
  ]
}}

CRITICAL INSTRUCTIONS FOR OUTPUT:
- Return ONLY valid JSON.
- DO NOT wrap the JSON in markdown code blocks.
- DO NOT add conversational text.

{format_instructions}"""
        ),
        HumanMessage(
            content=f"Topic: {state['topic']}\nGenerate the JSON plan now."
        )
    ])

    raw_response = model.invoke(prompt_value)
    
    try:
        plan = parser.parse(raw_response.content)
    except Exception:
        plan = retry_parser.parse_with_prompt(raw_response.content, prompt_value)
        
    return {"plan": plan}

def no_of_worker(state: State):
    return [
        Send(
            "Worker",
            {"task": task, "topic": state["topic"], "plan": state["plan"]}
        )
        for task in state["plan"].tasks
    ]

def worker(payload: dict) -> dict:
    task = payload['task']
    topic = payload['topic']
    plan = payload['plan']

    bullets_text = "\n- " + "\n- ".join(task.bullets) if task.bullets else ""

    section_md = model.invoke([
        SystemMessage(
            content="""You are a senior technical writer and developer advocate. Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to the Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).

Technical quality bar:
- Be precise and implementation-oriented.
- Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.

Markdown style:
- Start with a '## <Section Title>' heading.
- Use short paragraphs, bullet lists where helpful, and code fences for code.
- Avoid fluff. Avoid marketing language."""
        ),
        HumanMessage(
            content=(
                f"Blog: {plan.blog_title}\n"
                f"Audience: {plan.audience}\n"
                f"Tone: {plan.tone}\n"
                f"Topic: {topic}\n\n"
                f"Section: {task.title}\n"
                f"Section type: {task.section_type}\n"
                f"Goal: {task.goal}\n"
                f"Target words: {task.target_words}\n"
                f"Bullets:{bullets_text}\n"
                "Return only the section content in Markdown."
            )
        )
    ]).content.strip()

    return {'sections': [section_md]}

def reducer(state: State) -> dict:
    title = state['plan'].blog_title
    body = "\n\n".join(state['sections']).strip()

    result = f'# {title}\n\n{body}\n'

    filename = title.lower().replace(" ", "-") + ".md"
    output_path = Path(filename)
    output_path.write_text(result, encoding="utf-8")

    return {"result": result}

graph = StateGraph(State)

graph.add_node("Orchestrator", orchestrator)
graph.add_node("Worker", worker)
graph.add_node("Reducer", reducer)

graph.add_edge(START, "Orchestrator")
graph.add_conditional_edges("Orchestrator", no_of_worker, ["Worker"])
graph.add_edge("Worker", "Reducer")
graph.add_edge("Reducer", END)

app = graph.compile()

out = app.invoke({"topic": "Write a blog on Self Attention", "sections": []})
print(out['plan'])