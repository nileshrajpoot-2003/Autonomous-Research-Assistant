from __future__ import annotations

import json
import logging
import operator
import os
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/ as an SVG, e.g. qkv_flow.svg")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class ImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)


class ContentEvaluation(BaseModel):
    relevance: int = Field(..., ge=1, le=10)
    factual_accuracy: int = Field(..., ge=1, le=10)
    completeness: int = Field(..., ge=1, le=10)
    readability: int = Field(..., ge=1, le=10)
    structure: int = Field(..., ge=1, le=10)
    evidence_and_citations: int = Field(..., ge=1, le=10)
    technical_depth: int = Field(..., ge=1, le=10)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    claims_to_verify: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    final_remarks: str


EVALUATION_SCORE_FIELDS = (
    "relevance",
    "factual_accuracy",
    "completeness",
    "readability",
    "structure",
    "evidence_and_citations",
    "technical_depth",
)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    original_md: str
    evaluation: Optional[dict]
    initial_evaluation: Optional[dict]
    revision_attempts: int
    revision_applied: bool

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
openrouter_model = os.getenv("OPENROUTER_MODEL", "").strip()
openrouter_base_url = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
).strip().rstrip("/")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()

if openrouter_api_key and openrouter_model:
    llm_provider = "OpenRouter"
    llm_model = openrouter_model
    llm = ChatOpenAI(
        model=openrouter_model,
        base_url=openrouter_base_url,
        api_key=openrouter_api_key,
        max_tokens=10000,
    )
else:
    if openrouter_api_key or openrouter_model:
        logger.warning(
            "OpenRouter requires both OPENROUTER_API_KEY and OPENROUTER_MODEL; "
            "falling back to Ollama."
        )
    llm_provider = "Ollama"
    llm_model = ollama_model
    ollama_base_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    ).strip().rstrip("/")
    llm = ChatOpenAI(
        model=ollama_model,
        base_url=f"{ollama_base_url}/v1",
        api_key="ollama",
        max_tokens=10000,
    )

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY is not set; web research is skipped.")
        return []
    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=max_results)
        results = tool.invoke({"query": query})
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except json.JSONDecodeError:
                logger.warning("Tavily returned an unrecognized text response.")
                return []
        if isinstance(results, dict):
            if "error" in results:
                logger.warning("Tavily search failed: %s", results["error"])
                return []
            results = results.get("results", [results])
        if not isinstance(results, list):
            logger.warning(
                "Tavily returned an unsupported response type: %s",
                type(results).__name__,
            )
            return []

        out: List[dict] = []
        for r in results or []:
            if not isinstance(r, dict):
                logger.warning(
                    "Skipping a Tavily result with unsupported type: %s",
                    type(r).__name__,
                )
                continue
            title = r.get("title")
            url = r.get("url")
            snippet = r.get("content") or r.get("snippet")
            source = r.get("source")
            out.append(
                {
                    "title": title[:250] if isinstance(title, str) else "",
                    "url": url[:1000] if isinstance(url, str) else "",
                    "snippet": snippet[:400] if isinstance(snippet, str) else "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": source[:200] if isinstance(source, str) else None,
                }
            )
        return out
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Return no more than 12 evidence items.
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep each snippet under 400 characters; do not reproduce full page content.
- Deduplicate by URL.
"""

MAX_RESEARCH_QUERIES = 6
MAX_RESULTS_PER_QUERY = 3
MAX_RAW_RESULTS = 15
MAX_EVIDENCE_ITEMS = 12


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:MAX_RESEARCH_QUERIES]
    raw: List[dict] = []
    seen_urls = set()
    for q in queries:
        remaining = MAX_RAW_RESULTS - len(raw)
        if remaining <= 0:
            break
        results = _tavily_search(q, max_results=min(MAX_RESULTS_PER_QUERY, remaining))
        for result in results:
            url = result.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                raw.append(result)
                if len(raw) >= MAX_RAW_RESULTS:
                    break

    if not raw:
        return {"evidence": []}

    extractor = llm.bind(max_tokens=3000).with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{json.dumps(raw, ensure_ascii=False)}"
                )
            ),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())[:MAX_EVIDENCE_ITEMS]

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup: 
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) Content evaluation, revision, and image reduction (subgraph)
#    merge -> evaluate -> (revise once?) -> decide_images -> generate/place
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md, "original_md": merged_md}


EVALUATOR_SYSTEM = """You are an expert technical content evaluator.
Evaluate the complete blog against the user's topic, the planned sections, and
the supplied evidence. Score each criterion from 1 to 10:

- relevance: how directly the article addresses the topic
- factual_accuracy: technical correctness and whether claims are supported
- completeness: coverage of planned sections and important points
- readability: clarity, flow, and suitable language for the audience
- structure: useful headings, paragraphs, and organization
- evidence_and_citations: reliable, relevant source links for factual claims
- technical_depth: adequate explanations, examples, and technical detail

Do not treat the supplied evidence as proof for claims it does not support.
You cannot independently verify facts beyond the supplied evidence; list
uncertain, unsupported, or citation-sensitive claims in claims_to_verify.
Give specific, actionable strengths, weaknesses, and suggestions. Avoid
inflating scores. Return only the requested structured evaluation.
"""

REVISION_SYSTEM = """You are a careful technical editor revising a complete blog.
Use the evaluation feedback to improve the article. Preserve its topic,
intended audience, useful structure, and all planned sections. Correct or
qualify unsupported claims; do not invent facts or citations. Keep existing
valid source URLs attached to the claims they support. Return only the
complete revised article in Markdown, without commentary or code fences.
"""

EVALUATION_THRESHOLD = 7.0
MAX_REVISION_ATTEMPTS = 1


def _evaluation_context(state: State) -> tuple[Plan, list[dict]]:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Content evaluation called without a plan.")
    if isinstance(plan, dict):
        plan = Plan(**plan)

    evidence = []
    for item in state.get("evidence", []) or []:
        evidence_item = item if isinstance(item, EvidenceItem) else EvidenceItem(**item)
        evidence.append(
            {
                **evidence_item.model_dump(),
                "snippet": (evidence_item.snippet or "")[:600],
            }
        )
    return plan, evidence[:20]


def content_evaluator_node(state: State) -> dict:
    plan, evidence = _evaluation_context(state)
    content = state.get("merged_md", "").strip()
    if not content:
        raise ValueError("Content evaluation received an empty blog.")

    evaluator = llm.bind(max_tokens=3500).with_structured_output(ContentEvaluation)
    evaluation = evaluator.invoke(
        [
            SystemMessage(content=EVALUATOR_SYSTEM),
            HumanMessage(
                content=(
                    f"User topic: {state['topic']}\n"
                    f"Blog mode: {state.get('mode', 'closed_book')}\n"
                    f"Audience: {plan.audience}\n"
                    f"Planned sections and goals:\n"
                    f"{json.dumps([task.model_dump() for task in plan.tasks], ensure_ascii=False)}\n"
                    f"Available evidence:\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
                    f"Generated blog:\n{content}"
                )
            ),
        ]
    )
    report = evaluation.model_dump()
    report["overall_score"] = round(
        sum(report[field] for field in EVALUATION_SCORE_FIELDS)
        / len(EVALUATION_SCORE_FIELDS),
        1,
    )

    updates = {"evaluation": report}
    if not state.get("revision_attempts", 0):
        updates["initial_evaluation"] = report
    return updates


def route_after_evaluation(state: State) -> str:
    evaluation = state.get("evaluation")
    if evaluation is None:
        raise ValueError("Evaluation routing called without an evaluation report.")
    if (
        evaluation["overall_score"] < EVALUATION_THRESHOLD
        and state.get("revision_attempts", 0) < MAX_REVISION_ATTEMPTS
    ):
        return "revise"
    return "images"


def revise_content_node(state: State) -> dict:
    plan, evidence = _evaluation_context(state)
    evaluation = state.get("evaluation")
    if evaluation is None:
        raise ValueError("Content revision called without evaluation feedback.")

    response = llm.bind(max_tokens=10000).invoke(
        [
            SystemMessage(content=REVISION_SYSTEM),
            HumanMessage(
                content=(
                    f"User topic: {state['topic']}\n"
                    f"Audience: {plan.audience}\n"
                    f"Planned sections and goals:\n"
                    f"{json.dumps([task.model_dump() for task in plan.tasks], ensure_ascii=False)}\n"
                    f"Available evidence:\n{json.dumps(evidence, ensure_ascii=False)}\n"
                    f"Evaluation feedback:\n{json.dumps(evaluation, ensure_ascii=False)}\n\n"
                    f"Current blog:\n{state['merged_md']}"
                )
            ),
        ]
    )
    revised_content = response.content
    if not isinstance(revised_content, str) or not revised_content.strip():
        raise ValueError("The content revision model returned no Markdown text.")

    return {
        "merged_md": revised_content.strip(),
        "revision_attempts": state.get("revision_attempts", 0) + 1,
        "revision_applied": True,
    }


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Use descriptive .svg filenames; images are generated as vector diagrams.
- Avoid decorative images; prefer technical diagrams with short labels.
Return only image specs; do not rewrite or quote the blog.
"""

def decide_images(state: State) -> dict:
    planner = llm.bind(max_tokens=6000).with_structured_output(ImagePlan)
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n"
                    "Sections:\n"
                    + "\n".join(
                        f"- {task.title}: {task.goal}"
                        for task in plan.tasks
                    )
                )
            ),
        ]
    )

    image_specs = [
        {**img.model_dump(), "placeholder": f"[[IMAGE_{index}]]"}
        for index, img in enumerate(image_plan.images[:3], start=1)
    ]
    md_with_placeholders = state["merged_md"]
    if image_specs:
        image_section = "\n\n## Diagrams\n\n" + "\n\n".join(
            f"[[IMAGE_{index}]]" for index in range(1, len(image_specs) + 1)
        )
        md_with_placeholders = md_with_placeholders.rstrip() + image_section + "\n"

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": image_specs,
    }


def _generate_svg(prompt: str) -> bytes:
    """Generate and validate an SVG technical diagram with the configured model."""
    system_prompt = (
        "Create a clear, self-contained SVG technical diagram. "
        "Return only valid SVG XML, with a 1200x675 viewBox. Use readable "
        "labels, simple shapes, arrows, and a restrained color palette. "
        "Use only svg, g, rect, circle, ellipse, line, polyline, polygon, "
        "path, text, tspan, defs, and marker elements. Do not include "
        "scripts, styles, links, images, or non-SVG explanation."
    )
    allowed_elements = {
        "svg", "g", "rect", "circle", "ellipse", "line", "polyline",
        "polygon", "path", "text", "tspan", "defs", "marker",
    }
    allowed_attributes = {
        "viewBox", "width", "height", "x", "y", "x1", "y1", "x2", "y2",
        "cx", "cy", "r", "rx", "ry", "points", "d", "fill", "stroke",
        "stroke-width", "stroke-linecap", "stroke-linejoin", "stroke-dasharray",
        "opacity", "fill-opacity", "stroke-opacity", "transform", "font-size",
        "font-family", "font-weight", "text-anchor", "dominant-baseline",
        "id", "marker-end", "marker-start", "markerWidth", "markerHeight",
        "refX", "refY", "orient", "preserveAspectRatio",
    }
    validation_error = None
    for attempt in range(2):
        instruction = system_prompt
        if validation_error:
            instruction += (
                f" Your previous response was invalid ({validation_error}). "
                "Regenerate the entire diagram as well-formed SVG XML."
            )
        response = llm.invoke(
            [
                SystemMessage(content=instruction),
                HumanMessage(content=prompt),
            ]
        )
        svg_text = response.content
        if not isinstance(svg_text, str):
            raise ValueError("The local model did not return SVG text.")

        try:
            svg_match = re.search(r"<svg\b[\s\S]*?</svg\s*>", svg_text, flags=re.IGNORECASE)
            if not svg_match:
                raise ValueError("The response did not contain an SVG document.")
            svg_text = svg_match.group(0)
            if len(svg_text.encode("utf-8")) > 1_000_000:
                raise ValueError("The generated SVG exceeded the 1 MB size limit.")
            if re.search(r"<!DOCTYPE|<!ENTITY", svg_text, flags=re.IGNORECASE):
                raise ValueError("The generated SVG contains a forbidden XML declaration.")

            root = ET.fromstring(svg_text)
            if root.tag.rsplit("}", 1)[-1].lower() != "svg":
                raise ValueError("The generated document root is not SVG.")

            for element in root.iter():
                if element.tag.rsplit("}", 1)[-1].lower() not in allowed_elements:
                    raise ValueError("The generated SVG contains an unsupported element.")
                if any(
                    attribute.rsplit("}", 1)[-1] not in allowed_attributes
                    for attribute in element.attrib
                ):
                    raise ValueError("The generated SVG contains an unsupported attribute.")

            ET.register_namespace("", "http://www.w3.org/2000/svg")
            return ET.tostring(root, encoding="utf-8", xml_declaration=True)
        except (ET.ParseError, ValueError) as exc:
            validation_error = exc

    raise ValueError(f"The local model returned invalid SVG twice: {validation_error}") from validation_error


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        image_stem = Path(spec["filename"]).stem or "diagram"
        out_path = next(
            (
                images_dir / f"{image_stem}{extension}"
                for extension in (".svg", ".png", ".jpg", ".webp")
                if (images_dir / f"{image_stem}{extension}").exists()
            ),
            None,
        )
        if out_path is None:
            try:
                img_bytes = _generate_svg(spec["prompt"])
                filename = f"{image_stem}.svg"
                out_path = images_dir / filename
                out_path.write_bytes(img_bytes)
            except Exception as e:
                logger.exception("Image generation failed for %s", image_stem)
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        spec["filename"] = out_path.name
        img_md = f"![{spec['alt']}](images/{out_path.name})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("evaluate_content", content_evaluator_node)
reducer_graph.add_node("revise_content", revise_content_node)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "evaluate_content")
reducer_graph.add_conditional_edges(
    "evaluate_content",
    route_after_evaluation,
    {"revise": "revise_content", "images": "decide_images"},
)
reducer_graph.add_edge("revise_content", "evaluate_content")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
