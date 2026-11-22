from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st
from langchain_core.exceptions import ModelRateLimitError
from langchain_openai.chat_models.base import OpenAIConnectionError
from openai import LengthFinishReasonError

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from backend import app, content_evaluator_node, llm_model, llm_provider


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    latest_state = None
    for step in graph_app.stream(inputs, stream_mode="values"):
        latest_state = step
        yield ("values", step)

    if latest_state is None:
        raise RuntimeError("The workflow completed without returning a state.")
    yield ("final", latest_state)


def display_image_file(path: Path, caption: Optional[str] = None) -> None:
    if path.suffix.lower() != ".svg":
        st.image(str(path), caption=caption, width="stretch")
        return

    svg_text = path.read_text(encoding="utf-8")
    if len(svg_text.encode("utf-8")) > 1_000_000 or re.search(
        r"<!DOCTYPE|<!ENTITY", svg_text, flags=re.IGNORECASE
    ):
        raise ValueError(f"SVG file is oversized or contains a forbidden declaration: {path}")

    root = ET.fromstring(svg_text)
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        raise ValueError(f"Image file does not contain an SVG root element: {path}")

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    normalized_svg = ET.tostring(root, encoding="unicode")
    st.image(normalized_svg, caption=caption, width="stretch")


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), width="stretch")
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                display_image_file(img_path, caption=caption or (alt or None))
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# ✅ NEW: Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current working directory, newest first.
    Filters out obvious non-blog markdown files if needed.
    """
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Autonomous Research Assistant", page_icon="✍️", layout="wide")

st.title("Autonomous Research Assistant")
st.caption("Turn a topic into a researched, reviewed, and ready-to-publish article.")

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = []

with st.sidebar:
    past_files = list_past_blogs()
    with st.expander(f"Saved blogs ({len(past_files)})", expanded=False):
        if past_files:
            options: List[str] = []
            file_by_label: Dict[str, Path] = {}
            for path in past_files[:50]:
                try:
                    title = extract_title_from_md(read_md_file(path), path.stem)
                except OSError:
                    title = path.stem
                label = f"{title} · {path.name}"
                options.append(label)
                file_by_label[label] = path

            selected_label = st.selectbox(
                "Choose a saved blog",
                options,
                label_visibility="collapsed",
            )
            if st.button("Load blog", width="stretch"):
                selected_path = file_by_label[selected_label]
                md_text = read_md_file(selected_path)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.session_state["result_view"] = "✍️ Article"
        else:
            st.caption("Generated Markdown files will appear here.")

with st.container():
    with st.form("generate_blog", clear_on_submit=False):
        st.markdown("#### What should the article cover?")
        topic = st.text_area(
            "Topic",
            height=110,
            placeholder="For example: How retrieval-augmented generation works",
            label_visibility="collapsed",
        )
        with st.expander("Generation settings"):
            as_of = st.date_input("Research as of", value=date.today())
        run_btn = st.form_submit_button(
            "🚀 Generate article",
            type="primary",
            width="stretch",
        )

st.caption("Research, drafting, and quality evaluation run automatically.")

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "original_md": "",
        "evaluation": None,
        "initial_evaluation": None,
        "revision_attempts": 0,
        "revision_applied": False,
        "final": "",
    }

    status = st.status("Building your article…", expanded=False)

    current_state: Dict[str, Any] = {}
    last_node = None

    try:
        for kind, payload in try_stream(app, inputs):
            if kind in ("updates", "values"):
                node_name = None
                if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                    node_name = next(iter(payload.keys()))
                if node_name and node_name != last_node:
                    status.write(f"➡️ Node: `{node_name}`")
                    last_node = node_name

                current_state = extract_latest_state(current_state, payload)

                log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

            elif kind == "final":
                out = payload
                st.session_state["last_out"] = out
                st.session_state["result_view"] = "✍️ Article"
                status.update(label="✅ Done", state="complete", expanded=False)
                log("[final] received final state")
    except LengthFinishReasonError:
        status.update(
            label=f"Generation paused: {llm_provider} response limit",
            state="error",
            expanded=True,
        )
        st.error(
            "The model response reached its output limit. Research input is now "
            "bounded; retry, or use a narrower topic if the problem persists."
        )
        log(f"[{llm_provider.lower()}-length-limit] Structured model output was truncated.")
    except OpenAIConnectionError as exc:
        status.update(
            label=f"Generation paused: cannot reach {llm_provider}",
            state="error",
            expanded=True,
        )
        st.error(
            f"Could not connect to {llm_provider} for model `{llm_model}`. "
            "Check your internet connection, DNS, VPN, or proxy, then try again."
        )
        log(f"[{llm_provider.lower()}-connection-error] {exc}")
    except ModelRateLimitError as exc:
        status.update(
            label=f"Generation paused: {llm_provider} rate limit",
            state="error",
            expanded=True,
        )
        st.error(
            f"The {llm_provider} model `{llm_model}` is temporarily rate-limiting requests. "
            "Wait briefly, then try again."
        )
        log(f"[{llm_provider.lower()}-rate-limit] {exc}")

if logs:
    st.session_state["logs"].extend(logs)

# Render last result (if any)
out = st.session_state.get("last_out")
if (
    isinstance(out, dict)
    and out.get("final")
    and out.get("plan")
    and not isinstance(out.get("evaluation"), dict)
):
    plan_obj = out["plan"]
    plan_title = (
        plan_obj.blog_title
        if hasattr(plan_obj, "blog_title")
        else plan_obj.get("blog_title", "Saved blog")
    )
    evaluation_input = {
        "topic": out.get("topic") or plan_title,
        "mode": out.get("mode", "closed_book"),
        "plan": plan_obj,
        "evidence": out.get("evidence") or [],
        "merged_md": out["final"],
        "revision_attempts": 0,
    }
    try:
        evaluation_updates = content_evaluator_node(evaluation_input)
    except (OpenAIConnectionError, ModelRateLimitError):
        st.warning(
            "The current blog has no evaluation report yet. "
            "Check the model connection or rate limit, then generate again."
        )
    else:
        out.update(evaluation_updates)
        out["original_md"] = out["final"]
        out["revision_applied"] = False
        st.session_state["last_out"] = out

if isinstance(out, dict) and isinstance(out.get("evaluation"), dict):
    st.success(
        f"Content quality evaluation complete: "
        f"{out['evaluation']['overall_score']:.1f}/10. "
        "Open Quality for the full report."
    )

if out:
    plan_obj = out.get("plan")
    if hasattr(plan_obj, "model_dump"):
        plan_dict = plan_obj.model_dump()
    elif isinstance(plan_obj, dict):
        plan_dict = plan_obj
    else:
        plan_dict = {}

    evaluation = out.get("evaluation")
    evidence = out.get("evidence") or []
    final_md = out.get("final") or ""
    blog_title = plan_dict.get("blog_title") or extract_title_from_md(final_md, "Generated article")

    with st.container(border=True):
        summary_cols = st.columns([2, 1, 1])
        summary_cols[0].markdown(f"### {blog_title}")
        if isinstance(evaluation, dict):
            summary_cols[1].metric("Quality", f"{evaluation['overall_score']:.1f}/10")
        else:
            summary_cols[1].metric("Quality", "Pending")
        summary_cols[2].metric("Sources", len(evidence))

    result_view = st.radio(
        "Article sections",
        ["✍️ Article", "⭐ Quality", "🔎 Research", "🧩 Outline", "🖼️ Images", "⚙️ Activity"],
        horizontal=True,
        label_visibility="collapsed",
        key="result_view",
    )

    if result_view == "✍️ Article":
        if final_md:
            render_markdown_with_local_images(final_md)
            with st.expander("Download or export"):
                md_filename = f"{safe_slug(blog_title)}.md"
                download_cols = st.columns(2)
                download_cols[0].download_button(
                    "Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                    width="stretch",
                )
                download_cols[1].download_button(
                    "Download article bundle",
                    data=bundle_zip(final_md, md_filename, Path("images")),
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                    width="stretch",
                )
        else:
            st.info("No article content is available.")

    elif result_view == "⭐ Quality":
        st.subheader("Content quality")
        if not isinstance(evaluation, dict):
            st.info("This saved article does not have a quality report.")
        else:
            st.metric("Overall score", f"{evaluation['overall_score']:.1f} / 10")
            st.caption(
                "AI-assisted assessment, not an independent fact check. "
                "Review claims against their sources."
            )
            score_labels = {
                "relevance": "Relevance",
                "factual_accuracy": "Factual accuracy",
                "completeness": "Completeness",
                "readability": "Readability",
                "structure": "Structure",
                "evidence_and_citations": "Evidence & citations",
                "technical_depth": "Technical depth",
            }
            score_rows = [
                {"Criterion": label, "Score": evaluation[key], "Out of": 10}
                for key, label in score_labels.items()
            ]
            st.dataframe(pd.DataFrame(score_rows), width="stretch", hide_index=True)

            feedback_cols = st.columns(2)
            for col, label, key in (
                (feedback_cols[0], "Strengths", "strengths"),
                (feedback_cols[1], "Suggestions", "suggestions"),
                (feedback_cols[0], "Weaknesses", "weaknesses"),
                (feedback_cols[1], "Claims to verify", "claims_to_verify"),
            ):
                with col:
                    with st.expander(label, expanded=key == "claims_to_verify"):
                        items = evaluation.get(key) or []
                        if items:
                            st.markdown("\n".join(f"- {item}" for item in items))
                        else:
                            st.caption("None identified.")

            st.info(evaluation.get("final_remarks") or "No additional remarks.")
            if out.get("revision_applied"):
                initial_score = (out.get("initial_evaluation") or {}).get("overall_score")
                if initial_score is not None:
                    st.caption(
                        f"One revision applied: {initial_score:.1f}/10 → "
                        f"{evaluation['overall_score']:.1f}/10."
                    )
                with st.expander("Compare before and after revision"):
                    before, after = st.columns(2)
                    with before:
                        st.markdown("**Before**")
                        st.markdown(out.get("original_md") or "", unsafe_allow_html=False)
                    with after:
                        st.markdown("**After**")
                        st.markdown(final_md, unsafe_allow_html=False)

    elif result_view == "🔎 Research":
        st.subheader("Research sources")
        queries = out.get("queries") or []
        if queries:
            with st.expander(f"Search queries ({len(queries)})"):
                st.markdown("\n".join(f"- {query}" for query in queries))
        if evidence:
            rows = []
            for item in evidence:
                item_dict = item.model_dump() if hasattr(item, "model_dump") else item
                rows.append(
                    {
                        "Title": item_dict.get("title"),
                        "Published": item_dict.get("published_at"),
                        "Source": item_dict.get("source"),
                        "URL": item_dict.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No sources were needed or returned for this article.")

    elif result_view == "🧩 Outline":
        st.subheader("Article outline")
        if not plan_dict:
            st.info("No plan is available for this saved article.")
        else:
            plan_cols = st.columns(3)
            plan_cols[0].write(f"**Audience:** {plan_dict.get('audience', '—')}")
            plan_cols[1].write(f"**Tone:** {plan_dict.get('tone', '—')}")
            plan_cols[2].write(f"**Type:** {plan_dict.get('blog_kind', '—')}")
            tasks = plan_dict.get("tasks") or []
            if tasks:
                task_rows = [
                    {
                        "Section": task.get("title"),
                        "Target words": task.get("target_words"),
                        "Research": "Yes" if task.get("requires_research") else "No",
                        "Citations": "Yes" if task.get("requires_citations") else "No",
                    }
                    for task in tasks
                ]
                st.dataframe(pd.DataFrame(task_rows), width="stretch", hide_index=True)
                with st.expander("Section goals and details"):
                    st.json(tasks)
            constraints = plan_dict.get("constraints") or []
            if constraints:
                with st.expander("Writing constraints"):
                    st.markdown("\n".join(f"- {item}" for item in constraints))

    elif result_view == "🖼️ Images":
        st.subheader("Diagrams")
        specs = out.get("image_specs") or []
        if not specs:
            st.info("No diagrams were generated for this article.")
        else:
            with st.expander("Diagram plan"):
                st.json(specs)
            images_dir = Path("images")
            for spec in specs:
                image_path = images_dir / Path(spec.get("filename", "")).name
                if image_path.is_file():
                    display_image_file(image_path, caption=spec.get("caption") or image_path.name)
                else:
                    st.caption(f"Image file not found: {image_path.name}")
            with st.expander("Download images"):
                image_bundle = images_zip(images_dir)
                if image_bundle:
                    st.download_button(
                        "Download image bundle",
                        data=image_bundle,
                        file_name="images.zip",
                        mime="application/zip",
                        width="stretch",
                    )

    elif result_view == "⚙️ Activity":
        st.subheader("Generation activity")
        recent_logs = st.session_state.get("logs") or []
        if recent_logs:
            st.code("\n\n".join(recent_logs[-80:]), language="text")
        else:
            st.caption("No activity details are available.")
else:
    st.info("Enter a topic above to generate your first article.")
