# Image Generation feature added
from __future__ import annotations
import os
from datetime import date, timedelta

import operator
from typing import TypedDict,List,Annotated,Literal,Optional
from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel,Field
from langgraph.graph import StateGraph,START,END
from langgraph.types import Send

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser,RetryWithErrorOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()
class Task(BaseModel):
    id:int
    title:str
    goal:str=Field(
        description="One sentence describing what the reader should be able to do/understand after this section."
    )
    bullets:List[str]=Field(
        min_length=0,
        max_length=5,
        description="Upto 5 concrete, non-overlapping subpoints to cover in this section."
    )
    target_words:int=Field(
        description="Target word count for this section (120-450)."
    )
    # section_type:Literal[
    #     "intro","core","examples","checklist","common_mistakes","conclusion"
    # ]=Field(
    #     description="Use 'common_mistakes' exactly once in the plan."
    # )
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False
    # brief:str=Field(
    #     description="what to cover"
    # )
class Plan(BaseModel):
    blog_title:str
    audience:str=Field(description="Who this blog is for.")
    tone:str=Field(description="Writing tone (e.g., practical, crisp).")
    blog_kind:Literal["explainer","tutorial", "news_roundup", "comparison", "system_design"]="explainer"
    constraints:List[str]=Field(default_factory=list)
    tasks:list[Task]
class EvidenceItem(BaseModel):
    title:str
    url:str
    published_at:Optional[str]=None
    snippet:Optional[str]=None
    source:Optional[str]=None
class EvidencePack(BaseModel):
    evidence:List[EvidenceItem]=Field(default_factory=list)
class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)
class ImageSpec(BaseModel):
    placeholder:str=Field(description="e.g.[IMAGE-1]")
    filename:str=Field(description="Save under images/,e.g. qkv_flow.png")
    alt:str
    caption:str
    prompt:str=Field(description="Prompt to send to the image model.")
    size:Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality:Literal["low", "medium", "high"] = "medium"
class GlobalImagePlan(BaseModel):
    md_with_placeholders:str
    images:List[ImageSpec]=Field(default_factory=list)
class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    result: str
    # topic:str
    # plan:Optional[Plan]
    # sections:Annotated[List[str],operator.add]
    # result:str
    # mode:str
    # need_research:bool
    # queries:List[str]
    # evidence:List[EvidenceItem]
    

# Yeh setup top par rakho
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # Model change karna zaruri hai
    task="text-generation",
    max_new_tokens=2500,
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)
parser = PydanticOutputParser(pydantic_object=Plan)
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)
# -----------------------------
# 3) Router (decide upfront)
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3-10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

router_parser = PydanticOutputParser(
    pydantic_object=RouterDecision
)

router_retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=router_parser,
    llm=model
)

import json
import re

def router(state: State) -> dict:
    topic = state["topic"]

    prompt_value = ChatPromptValue(
        messages=[
            SystemMessage(
                content=f"""
{ROUTER_SYSTEM}

CRITICAL RULES:
- Return ONLY a JSON object.
- No markdown.
- No explanations.
- No text before JSON.
- No text after JSON.

Example:

{{
  "needs_research": true,
  "mode": "hybrid",
  "queries": [
    "latest multimodal llm benchmarks 2026",
    "state of multimodal llms 2026"
  ]
}}
"""
            ),
            HumanMessage(
                content=f"Topic: {topic}"
            )
        ]
    )

    response = model.invoke(prompt_value)

    raw = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    ).strip()

    try:
        decision = router_parser.parse(raw)

    except Exception as first_error:

        try:
            decision = router_retry_parser.parse_with_prompt(
                raw,
                prompt_value
            )

        except Exception:

            match = re.search(
                r"\{.*\}",
                raw,
                re.DOTALL
            )

            if not match:
                raise ValueError(
                    f"Could not find JSON in router output:\n{raw}"
                ) from first_error

            try:
                data = json.loads(match.group(0))
                decision = RouterDecision(**data)

            except Exception as json_error:
                raise ValueError(
                    f"Invalid router JSON:\n{raw}"
                ) from json_error

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def router_next(state:State)->str:
    return "research" if state["needs_research"] else "orchestrator"

def tavily_search(query: str, max_result: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_result)

    try:
        results = tool.invoke({"query": query})
    except Exception:
        return []

    normalized: List[dict] = []

    for i in results or []:
        normalized.append(
            {
                "title": i.get("title") or "",
                "url": i.get("url") or "",
                "snippet": i.get("content") or i.get("snippet") or "",
                "published_at": i.get("published_date") or i.get("published_at"),
                "source": i.get("source"),
            }
        )

    return normalized

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""
research_parser = PydanticOutputParser(
    pydantic_object=EvidencePack
)

research_retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=research_parser,
    llm=model
)
def research_node(state: State) -> dict:
    queries = state.get("queries", []) or []

    raw_results = []

    # Keep search volume reasonable
    for q in queries[:4]:
        raw_results.extend(
            tavily_search(q, 3)
        )

    if not raw_results:
        return {
            "evidence": []
        }

    evidence = []
    seen_urls = set()

    for item in raw_results:

        url = item.get("url")

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        evidence.append(
            EvidenceItem(
                title=item.get("title", ""),
                url=url,
                snippet=item.get("snippet"),
                published_at=item.get("published_at"),
                source=item.get("source"),
            )
        )

    return {
        "evidence": evidence[:20]
    }

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5-9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3-6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120-550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""
plan_parser = PydanticOutputParser(
    pydantic_object=Plan
)

plan_retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=plan_parser,
    llm=model
)

def orchestrator_node(state: State) -> dict:
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    response = model.invoke(
        [
            SystemMessage(
                content=(
                    ORCH_SYSTEM
                    + "\n\n"
                    + plan_parser.get_format_instructions()
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence:\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    plan = plan_parser.parse(response.content)

    return {"plan": plan}


# def orchestrator_node(state: State) -> dict:
#     import json
#     import re

#     evidence = state.get("evidence", [])
#     mode = state.get("mode", "closed_book")

#     messages = [
#         SystemMessage(
#             content=(
#                 ORCH_SYSTEM
#                 + "\n\n"
#                 + plan_parser.get_format_instructions()
#             )
#         ),
#         HumanMessage(
#             content=(
#                 f"Topic: {state['topic']}\n"
#                 f"Mode: {mode}\n\n"
#                 f"Evidence:\n"
#                 f"{[e.model_dump() for e in evidence][:16]}"
#             )
#         ),
#     ]

#     response = model.invoke(messages)

#     raw = (
#         response.content
#         if hasattr(response, "content")
#         else str(response)
#     ).strip()

#     print("\n===== ORCHESTRATOR RAW =====")
#     print(raw)
#     print("============================\n")

#     try:
#         plan = plan_parser.parse(raw)

#     except Exception:

#         match = re.search(
#             r"```(?:json)?\s*(\{.*?\})\s*```",
#             raw,
#             re.DOTALL,
#         )

#         if match:
#             json_text = match.group(1)
#         else:
#             start = raw.find("{")
#             end = raw.rfind("}")

#             if start == -1 or end == -1:
#                 raise ValueError(
#                     f"Could not find JSON in orchestrator output:\n{raw}"
#                 )

#             json_text = raw[start:end + 1]

#         print("\n===== EXTRACTED JSON =====")
#         print(json_text)
#         print("==========================\n")

#         try:
#             data = json.loads(json_text)
#             plan = Plan(**data)

#         except Exception as json_error:
#             raise ValueError(
#                 f"Invalid orchestrator JSON:\n{json_text}"
#             ) from json_error

#     print("\n===== PLAN CREATED =====")
#     print(plan.blog_title)
#     print(f"Tasks: {len(plan.tasks)}")
#     print("========================\n")

#     return {
#         "plan": plan
#     }


# def orchestrator_node(state: State) -> dict:
#     import json
#     import re

#     evidence = state.get("evidence", [])
#     mode = state.get("mode", "closed_book")

#     messages = [
#         SystemMessage(
#             content=(
#                 ORCH_SYSTEM
#                 + "\n\n"
#                 + plan_parser.get_format_instructions()
#             )
#         ),
#         HumanMessage(
#             content=(
#                 f"Topic: {state['topic']}\n"
#                 f"Mode: {mode}\n\n"
#                 f"Evidence:\n"
#                 f"{[e.model_dump() for e in evidence][:16]}"
#             )
#         ),
#     ]

#     response = model.invoke(messages)

#     raw = (
#         response.content
#         if hasattr(response, "content")
#         else str(response)
#     ).strip()

#     print("\n===== ORCHESTRATOR RAW =====")
#     print(raw)
#     print("============================\n")

#     try:
#         plan = plan_parser.parse(raw)

#     except Exception:

#         match = re.search(
#             r"\{.*\}",
#             raw,
#             re.DOTALL
#         )

#         if not match:
#             raise ValueError(
#                 f"Could not find JSON in orchestrator output:\n{raw}"
#             )

#         try:
#             data = json.loads(match.group(0))
#             plan = Plan(**data)

#         except Exception as json_error:
#             raise ValueError(
#                 f"Invalid orchestrator JSON:\n{raw}"
#             ) from json_error

#     print("\n===== PLAN CREATED =====")
#     print(plan.blog_title)
#     print(f"Tasks: {len(plan.tasks)}")
#     print("========================\n")

#     return {
#         "plan": plan
#     }

# def orchestrator_node(state: State) -> dict:
#     import json
#     import re

#     evidence = state.get("evidence", [])
#     mode = state.get("mode", "closed_book")

#     messages = [
#         SystemMessage(
#             content=(
#                 ORCH_SYSTEM
#                 + "\n\n"
#                 + plan_parser.get_format_instructions()
#             )
#         ),
#         HumanMessage(
#             content=(
#                 f"Topic: {state['topic']}\n"
#                 f"Mode: {mode}\n\n"
#                 f"Evidence:\n"
#                 f"{[e.model_dump() for e in evidence][:16]}"
#             )
#         ),
#     ]

#     response = model.invoke(messages)

#     raw = (
#         response.content
#         if hasattr(response, "content")
#         else str(response)
#     ).strip()

#     print("\n===== ORCHESTRATOR RAW =====")
#     print(raw)
#     print("============================\n")

#     try:
#         plan = plan_parser.parse(raw)

#     except Exception as first_error:

#         try:
#             plan = plan_retry_parser.parse_with_prompt(
#                 raw,
#                 messages
#             )

#         except Exception:

#             match = re.search(
#                 r"\{.*\}",
#                 raw,
#                 re.DOTALL
#             )

#             if not match:
#                 raise ValueError(
#                     f"Could not find JSON in orchestrator output:\n{raw}"
#                 ) from first_error

#             try:
#                 data = json.loads(match.group(0))
#                 plan = Plan(**data)

#             except Exception as json_error:
#                 raise ValueError(
#                     f"Invalid orchestrator JSON:\n{raw}"
#                 ) from json_error

#     print("\n===== PLAN CREATED =====")
#     print(plan.blog_title)
#     print(f"Tasks: {len(plan.tasks)}")
#     print("========================\n")

#     return {
#         "plan": plan
#     }

# def orchestrator_node(state: State) -> dict:
#     evidence = state.get("evidence", [])
#     mode = state.get("mode", "closed_book")

#     response = model.invoke(
#         [
#             SystemMessage(
#                 content=(
#                     ORCH_SYSTEM
#                     + "\n\n"
#                     + plan_parser.get_format_instructions()
#                 )
#             ),
#             HumanMessage(
#                 content=(
#                     f"Topic: {state['topic']}\n"
#                     f"Mode: {mode}\n\n"
#                     f"Evidence:\n"
#                     f"{[e.model_dump() for e in evidence][:16]}"
#                 )
#             ),
#         ]
#     )

#     plan = plan_parser.parse(response.content)

#     return {"plan": plan}

def no_of_worker(state:State):
    return [
        Send(
            "worker",
            {
                "task":task.model_dump(),
                "topic":state["topic"],
                "plan":state["plan"].model_dump(),
                "mode":state["mode"],
                "evidence":[e.model_dump() for e in state.get("evidence",[])]
            },
        )
        for task in state["plan"].tasks
    ]
# -----------------------------
# 7) Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
            for e in evidence[:20]
        )

    response = model.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n"
                    f"{evidence_text}\n"
                )
            ),
        ]
    )

    section_md = response.content.strip()

    return {
        "sections": [(task.id, section_md)]
    }
# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}
import json
import re

image_parser = PydanticOutputParser(
    pydantic_object=GlobalImagePlan
)

image_retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=image_parser,
    llm=llm
)
DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:

    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    prompt = f"""
{DECIDE_IMAGES_SYSTEM}

{image_parser.get_format_instructions()}

Blog kind: {plan.blog_kind}
Topic: {state['topic']}

Insert placeholders + propose image prompts.

{merged_md}

Return ONLY valid JSON.
"""

    response = llm.invoke(prompt)

    text = response.content if hasattr(response, "content") else str(response)

    try:
        image_plan = image_retry_parser.parse(text)

    except Exception:

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError(
                f"No valid JSON found in image planner output:\n{text}"
            )

        data = json.loads(match.group())
        image_plan = GlobalImagePlan(**data)

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }



def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def generate_and_place_images(state: State) -> dict:

    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{plan.blog_title}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"result": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"result": md}

reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

reducer_subgraph
# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", router_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", no_of_worker, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app

# -----------------------------
# 10) Runner
# -----------------------------
def run(topic: str, as_of: Optional[str] = None):
    if as_of is None:
        as_of = date.today().isoformat()

    out = app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of,
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "result": "",
        }
    )

    return out

