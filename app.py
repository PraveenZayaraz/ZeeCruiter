"""Streamlit interface for explainable multi-role resume matching."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ollama import OllamaClient
from src.parsers import DocumentParsingError, parse_pdf, parse_text
from src.reports import csv_report, json_report, markdown_report
from src.scoring import HybridScorer


EXAMPLE_ROLES = {
    "Backend Engineer": """Build Python services using FastAPI and PostgreSQL. Develop REST APIs,
write automated tests, use Docker and Git, and deploy workloads on AWS. Experience with Redis,
CI/CD, and Kubernetes is valuable.""",
    "Machine Learning Engineer": """Develop production machine learning and NLP systems using Python,
PyTorch, scikit-learn, Pandas, and NumPy. Work with Docker, AWS, REST APIs, and large language models.""",
    "Frontend Engineer": """Build accessible web interfaces using JavaScript, TypeScript, React, HTML,
and CSS. Collaborate through Git and Agile practices; experience with testing and REST APIs is preferred.""",
}


def _uploaded_text(uploaded) -> tuple[str, list[str]]:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".pdf":
        parsed = parse_pdf(uploaded)
        return parsed.text, list(parsed.warnings)
    return parse_text(uploaded), []


def _chart(results):
    names = [result.role_title for result in results]
    figure = go.Figure()
    figure.add_bar(name="Overall", x=names, y=[result.overall_score for result in results])
    for component in ("semantic", "lexical", "skills"):
        figure.add_bar(
            name=component.title(),
            x=names,
            y=[result.component_scores[component] or 0 for result in results],
        )
    figure.update_layout(
        barmode="group",
        yaxis_title="Textual alignment score",
        yaxis_range=[0, 100],
        legend_title_text="Component",
        height=430,
    )
    return figure


@st.cache_resource(show_spinner=False)
def _scorer() -> HybridScorer:
    return HybridScorer()


st.set_page_config(page_title="ZeeCruiter", page_icon="🎯", layout="wide")
st.title("ZeeCruiter")
st.caption("Explainable resume-to-role matching with optional private, local AI feedback")
st.info(
    "Scores represent textual alignment—not hiring probability. Candidate documents are processed in memory and are not saved."
)

with st.sidebar:
    st.header("Analysis settings")
    use_ollama = st.checkbox("Generate optional Ollama feedback", value=False)
    st.caption("The deterministic score never uses LLM output.")
    if use_ollama:
        status = OllamaClient().health()
        st.write(status.message)

resume_file = st.file_uploader("1. Upload a resume", type=["pdf"])

st.subheader("2. Add job descriptions")
input_mode = st.radio(
    "Input method", ["Paste roles", "Upload files", "Use examples"], horizontal=True
)
jobs: list[dict[str, str]] = []

if input_mode == "Paste roles":
    role_count = st.number_input("Number of roles", min_value=1, max_value=5, value=1)
    for index in range(int(role_count)):
        left, right = st.columns([1, 3])
        title = left.text_input("Role title", value=f"Role {index + 1}", key=f"title-{index}")
        text = right.text_area("Job description", height=140, key=f"jd-{index}")
        if text.strip():
            jobs.append({"id": str(index + 1), "title": title.strip() or f"Role {index + 1}", "text": text})
elif input_mode == "Upload files":
    jd_files = st.file_uploader(
        "Upload up to five PDF or UTF-8 TXT job descriptions",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    for index, file in enumerate(jd_files[:5]):
        try:
            text, warnings = _uploaded_text(file)
            for warning in warnings:
                st.warning(f"{file.name}: {warning}")
            if text:
                jobs.append({"id": str(index + 1), "title": Path(file.name).stem, "text": text})
        except DocumentParsingError as error:
            st.error(f"{file.name}: {error}")
else:
    selected = st.multiselect(
        "Example roles", list(EXAMPLE_ROLES), default=list(EXAMPLE_ROLES)
    )
    jobs = [
        {"id": str(index + 1), "title": title, "text": EXAMPLE_ROLES[title]}
        for index, title in enumerate(selected)
    ]

if st.button("Analyze alignment", type="primary", use_container_width=True):
    if resume_file is None:
        st.error("Upload a resume before running the analysis.")
    elif not jobs:
        st.error("Add at least one non-empty job description.")
    else:
        try:
            parsed_resume = parse_pdf(resume_file)
            for warning in parsed_resume.warnings:
                st.warning(warning)
            if not parsed_resume.text:
                st.stop()
            with st.spinner("Analyzing role alignment..."):
                results = _scorer().evaluate_many(parsed_resume.text, jobs)
            st.session_state["results"] = results
            st.session_state["resume_text"] = parsed_resume.text
            st.session_state["jobs"] = jobs
        except DocumentParsingError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Analysis could not be completed safely ({error.__class__.__name__}).")

results = st.session_state.get("results", [])
if results:
    st.subheader("Role comparison")
    table = pd.DataFrame(
        [
            {
                "Role": result.role_title,
                "Overall": result.overall_score,
                "Semantic": result.component_scores["semantic"],
                "Lexical": result.component_scores["lexical"],
                "Skill coverage": result.component_scores["skills"],
                "Matched skills": len(result.matched_skills),
                "Missing skills": len(result.missing_skills),
            }
            for result in results
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.plotly_chart(_chart(results), use_container_width=True)

    for result in results:
        with st.expander(f"{result.role_title} — {result.overall_score:.2f}/100"):
            st.write(result.semantic_message)
            if any(result.semantic_truncated.values()):
                st.warning("A document exceeded the configured semantic chunk limit; the report records which document was truncated.")
            st.caption(f"Effective weights: {result.effective_weights}")
            left, right = st.columns(2)
            left.markdown("**Matched skills**")
            left.write(", ".join(result.matched_skills) or "None detected")
            right.markdown("**Missing required skills**")
            right.write(", ".join(result.missing_required_skills) or "None detected")
            right.markdown("**Missing preferred skills**")
            right.write(", ".join(result.missing_preferred_skills) or "None detected")
            if result.matched_skills:
                st.markdown("**Why each skill matched**")
                for skill in result.matched_skills:
                    with st.container(border=True):
                        st.markdown(f"**{skill}**")
                        st.caption(f"Résumé evidence: {result.resume_evidence[skill][0]}")
                        st.caption(f"Job evidence: {result.job_evidence[skill][0]}")
            if use_ollama:
                job_text = next(job["text"] for job in st.session_state["jobs"] if job["id"] == result.role_id)
                with st.spinner(f"Generating local feedback for {result.role_title}..."):
                    st.markdown(OllamaClient().feedback(st.session_state["resume_text"], job_text, result))

    st.subheader("Download report")
    json_col, csv_col, md_col = st.columns(3)
    json_col.download_button(
        "Download JSON", json_report(results), "zecruiter_report.json", "application/json", use_container_width=True
    )
    csv_col.download_button(
        "Download CSV", csv_report(results), "zecruiter_report.csv", "text/csv", use_container_width=True
    )
    md_col.download_button(
        "Download Markdown", markdown_report(results), "zecruiter_report.md", "text/markdown", use_container_width=True
    )
