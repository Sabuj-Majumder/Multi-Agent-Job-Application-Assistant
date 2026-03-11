"""Streamlit UI for the Multi-Agent Job Application Assistant.

Run with: streamlit run app.py
"""

import csv
import io
import json
from typing import Any, Dict, List, Optional

import streamlit as st
from pypdf import PdfReader

from graph import pipeline
from utils.state import AgentState, CandidateProfile, Job

# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Job Application Assistant",
        layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialization ─────────────────────────────────────────────

if "pipeline_result"not in st.session_state:
    st.session_state["pipeline_result"] = None
if "resume_text"not in st.session_state:
    st.session_state["resume_text"] = None
if "has_run"not in st.session_state:
    st.session_state["has_run"] = False


# ── Helper Functions ─────────────────────────────────────────────────────────


def extract_pdf_text(uploaded_file: Any) -> str:
    """Extract text content from an uploaded PDF file.

    Args:
        uploaded_file: Streamlit UploadedFile object containing a PDF.

    Returns:
        Extracted text content from all pages of the PDF.
    """
    reader = PdfReader(uploaded_file)
    text_parts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def filter_jobs(
    jobs: List[Dict[str, Any]],
    keyword: str = "",
    sources: Optional[List[str]] = None,
    min_score: int = 0,
) -> List[Dict[str, Any]]:
    """Filter job results by keyword, source, and minimum fit score.

    Args:
        jobs: List of job dictionaries to filter.
        keyword: Keyword to search in title, company, and description.
        sources: List of source names to include. None means all.
        min_score: Minimum fit score to include. Defaults to 0.

    Returns:
        Filtered list of job dictionaries.
    """
    filtered: List[Dict[str, Any]] = []
    keyword_lower = keyword.lower().strip()

    for job in jobs:
        # Handle both Job objects and dicts
        if isinstance(job, Job):
            job_dict = job.model_dump()
        elif isinstance(job, dict):
            job_dict = job
        else:
            continue

        # Keyword filter
        if keyword_lower:
            searchable = (
                job_dict.get("title", "").lower()
                + " "
                + job_dict.get("company", "").lower()
                + " "
                + job_dict.get("description", "").lower()
            )
            if keyword_lower not in searchable:
                continue

        # Source filter
        if sources and job_dict.get("source", "") not in [s.lower() for s in sources]:
            continue

        # Fit score filter
        fit_score = job_dict.get("fit_score")
        if fit_score is not None and fit_score < min_score:
            continue

        filtered.append(job_dict)

    return filtered


def jobs_to_csv(jobs: List[Dict[str, Any]]) -> str:
    """Convert job list to CSV string for download.

    Args:
        jobs: List of job dictionaries.

    Returns:
        CSV-formatted string with headers.
    """
    output = io.StringIO()
    fieldnames = ["title", "company", "location", "salary", "url", "source", "fit_score"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for job in jobs:
        if isinstance(job, Job):
            job = job.model_dump()
        writer.writerow({k: job.get(k, "") for k in fieldnames})
    return output.getvalue()


def jobs_to_json(jobs: List[Dict[str, Any]]) -> str:
    """Convert job list to JSON string for download.

    Args:
        jobs: List of job dictionaries.

    Returns:
        Pretty-printed JSON string.
    """
    serializable = []
    for job in jobs:
        if isinstance(job, Job):
            serializable.append(job.model_dump())
        elif isinstance(job, dict):
            serializable.append(job)
    return json.dumps(serializable, indent=2, default=str)


def get_score_color(score: int) -> str:
    """Get display color for a fit score value.

    Args:
        score: Fit score integer 0-100.

    Returns:
        Color name: "green"for ≥70, "orange"for 40-69, "red"for <40.
    """
    if score >= 70:
        return "green"
    elif score >= 40:
        return "orange"
    return "red"


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Agent Pipeline")

    # Agent status table
    agents_info = [
        ("Job Scraper", "Fetches from 3 APIs"),
        ("Resume Analyzer", "Extracts profile from PDF"),
        ("Fit Scorer", "Scores job-candidate fit (v2)"),
        ("Resume Tailor", "Rewrites resume bullets (v2)"),
        ("Cover Letter", "Generates cover letters (v2)"),
    ]

    completed = []
    if st.session_state.get("pipeline_result"):
        completed = st.session_state["pipeline_result"].get("completed_agents", [])

    for agent_name, agent_desc in agents_info:
        agent_key = agent_name.split(" ", 1)[1].lower().replace(" ", "_")
        if agent_key in completed:
            status = "Complete"
        else:
            status = "Pending"
        st.markdown(f"**{agent_name}** — {status}")
        st.caption(agent_desc)

    st.divider()

    # Free APIs section
    st.subheader("Free APIs Used")
    st.markdown(
        """
    - [Groq](https://console.groq.com) — Free LLM inference
    - [Adzuna](https://developer.adzuna.com) — Job listings (free tier)
    - [RemoteOK](https://remoteok.com/api) — Remote jobs (no key needed)
    - [Jooble](https://jooble.org/api/index) — Job aggregation (free tier)
    """
    )

    st.divider()

    # How it works
    st.subheader("How It Works")
    st.markdown(
        """
    1. Enter a job title and location
    2. Our agents scrape jobs from 3 free APIs
    3. Upload your resume for AI-powered analysis
    """
    )

    st.divider()
    st.caption("Built with LangGraph · Groq · Streamlit")
    st.caption("[GitHub Repository](#)")


# ── Main Area — 4 Tabs ──────────────────────────────────────────────────────

st.title("Job Application Assistant")
st.caption("Multi-agent AI system for autonomous job searching and resume analysis")

tab_search, tab_results, tab_tailored, tab_export = st.tabs(
    ["Search", "Results", "Tailored Materials (v2)", "Export"]
)

# ── Tab 1: Search ────────────────────────────────────────────────────────────

with tab_search:
    st.subheader("Configure Your Job Search")

    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input(
            "Job Title",
            placeholder="AI Engineer, Data Scientist...",
            help="Enter the job title you're searching for",
        )
    with col2:
        location = st.text_input(
            "Location",
            placeholder="Remote, New York, London",
            help="Enter location or 'Remote'for remote jobs",
        )

    num_results = st.number_input(
        "Results per source",
        min_value=5,
        max_value=50,
        value=10,
        help="Number of jobs to fetch from each API source",
    )

    st.divider()

    # Resume upload
    st.subheader("Resume Upload (Optional)")
    uploaded_file = st.file_uploader(
        "Upload your resume for fit scoring (optional)",
        type=["pdf"],
        help="Upload a PDF resume for AI-powered analysis",
    )

    if uploaded_file is not None:
        try:
            resume_text = extract_pdf_text(uploaded_file)
            st.session_state["resume_text"] = resume_text
            word_count = len(resume_text.split())
            st.success(f"Resume uploaded successfully! ({word_count} words extracted)")
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {str(e)}")
            st.session_state["resume_text"] = None

    st.divider()

    # Run pipeline button
    if st.button("Run Agent Pipeline", type="primary", use_container_width=True):
        if not job_title:
            st.warning("Please enter a job title to search.")
        elif not location:
            st.warning("Please enter a location.")
        else:
            with st.status("Agents working...", expanded=True) as status:
                st.write("Job Scraper Agent — fetching from Adzuna, RemoteOK, Jooble...")

                # Build initial state
                initial_state: Dict[str, Any] = {
                    "job_title": job_title,
                    "location": location,
                    "num_results": int(num_results),
                    "resume_text": st.session_state.get("resume_text"),
                    "raw_jobs": [],
                    "jobs": [],
                    "scrape_summary": "",
                    "candidate_profile": None,
                    "ranked_jobs": [],
                    "tailored_bullets": None,
                    "cover_letters": None,
                    "error": None,
                    "active_agent": "job_scraper",
                    "completed_agents": [],
                }

                try:
                    result = pipeline.invoke(initial_state)

                    # Convert Job objects for display if needed
                    jobs_data = result.get("jobs", [])
                    job_count = len(jobs_data)

                    st.write(f"Job Scraper complete — found {job_count} jobs")

                    if result.get("resume_text"):
                        st.write("Resume Analyzer — extracting profile...")
                        if result.get("candidate_profile"):
                            profile = result["candidate_profile"]
                            if isinstance(profile, CandidateProfile):
                                skill_count = len(profile.skills)
                            elif isinstance(profile, dict):
                                skill_count = len(profile.get("skills", []))
                            else:
                                skill_count = 0
                            st.write(f"Resume Analyzer complete — found {skill_count} skills")

                    st.session_state["pipeline_result"] = result
                    st.session_state["has_run"] = True

                    status.update(label="Pipeline complete!", state="complete")

                except Exception as e:
                    status.update(label="Pipeline failed", state="error")
                    st.error(f"Pipeline execution failed: {str(e)}")

# ── Tab 2: Results ───────────────────────────────────────────────────────────

with tab_results:
    if not st.session_state.get("has_run"):
        st.info("Run a search first to see results here.")
    else:
        result = st.session_state["pipeline_result"]

        # Show errors if any
        if result and result.get("error"):
            st.error(result["error"])

        # Summary banner
        if result and result.get("scrape_summary"):
            st.info(result["scrape_summary"])

        # Get jobs list
        jobs_list = result.get("jobs", []) if result else []

        if jobs_list:
            # Filters
            st.subheader("Filters")
            filter_col1, filter_col2, filter_col3 = st.columns(3)

            with filter_col1:
                keyword_filter = st.text_input(
                    "Keyword filter",
                    placeholder="Search title, company, description...",
                    key="keyword_filter",
                )

            with filter_col2:
                source_options = list(set(
                    (j.source if isinstance(j, Job) else j.get("source", ""))
                    for j in jobs_list
                ))
                source_filter = st.multiselect(
                    "Source filter",
                    options=source_options,
                    default=source_options,
                    key="source_filter",
                )

            with filter_col3:
                # Check if any job has fit_score
                has_scores = any(
                    (j.fit_score if isinstance(j, Job) else j.get("fit_score"))
                    is not None
                    for j in jobs_list
                )
                if has_scores:
                    min_score = st.slider(
                        "Minimum fit score",
                        min_value=0,
                        max_value=100,
                        value=0,
                        key="min_score",
                    )
                else:
                    min_score = 0

            # Apply filters
            filtered_jobs = filter_jobs(
                jobs_list,
                keyword=keyword_filter,
                sources=source_filter,
                min_score=min_score,
            )

            st.caption(f"Showing {len(filtered_jobs)} of {len(jobs_list)} jobs")

            # Job cards
            for job in filtered_jobs:
                if isinstance(job, Job):
                    job_dict = job.model_dump()
                else:
                    job_dict = job

                title = job_dict.get("title", "Unknown")
                company = job_dict.get("company", "Unknown")
                location_str = job_dict.get("location", "Unknown")

                with st.expander(f"**{title}** — {company} · {location_str}"):
                    left_col, right_col = st.columns([3, 1])

                    with left_col:
                        st.markdown(f"**Company:** {company}")
                        st.markdown(f"**Location:** {location_str}")

                        salary = job_dict.get("salary")
                        if salary:
                            st.markdown(f"**Salary:** {salary}")

                        tags = job_dict.get("tags", [])
                        if tags:
                            tag_display = " ".join([f"`{tag}`"for tag in tags[:10]])
                            st.markdown(f"**Tags:** {tag_display}")

                    with right_col:
                        source = job_dict.get("source", "unknown")
                        st.markdown(f"**Source:** `{source}`")

                        url = job_dict.get("url", "")
                        if url:
                            st.link_button("View Job →", url)

                    # Fit score display
                    fit_score = job_dict.get("fit_score")
                    if fit_score is not None:
                        color = get_score_color(fit_score)
                        st.progress(fit_score / 100)
                        st.markdown(
                            f"**Fit Score:** :{color}[{fit_score}/100]"
                        )
                        reasoning = job_dict.get("fit_reasoning", "")
                        if reasoning:
                            st.caption(reasoning)

                    # Description
                    description = job_dict.get("description", "")
                    if len(description) > 400:
                        st.markdown(description[:400] + "...")
                    elif description:
                        st.markdown(description)
        else:
            st.warning("No jobs found. Try adjusting your search criteria.")

# ── Tab 3: Tailored Materials (v2) ───────────────────────────────────────────

with tab_tailored:
    if not st.session_state.get("has_run"):
        st.info("Run a search first to see tailored materials here.")
    else:
        result = st.session_state.get("pipeline_result")

        tailored_bullets = result.get("tailored_bullets") if result else None
        cover_letters = result.get("cover_letters") if result else None

        if not tailored_bullets and not cover_letters:
            st.info(
                "Tailored materials will be available in v2. "
                "This feature will rewrite your resume bullets and generate "
                "personalized cover letters for your top job matches."
            )
        else:
            ranked_jobs = result.get("ranked_jobs", [])
            top_jobs = ranked_jobs[:3] if ranked_jobs else []

            for i, job in enumerate(top_jobs):
                if isinstance(job, Job):
                    job_dict = job.model_dump()
                else:
                    job_dict = job

                job_id = job_dict.get("id", "")
                title = job_dict.get("title", "Unknown")
                company = job_dict.get("company", "Unknown")

                with st.expander(f"**{i+1}. {title}** — {company}", expanded=(i == 0)):
                    # Tailored bullets
                    if tailored_bullets and job_id in tailored_bullets:
                        st.subheader("Tailored Resume Bullets")
                        bullets = tailored_bullets[job_id]
                        for j, bullet in enumerate(bullets, 1):
                            st.markdown(f"{j}. {bullet}")
                        if st.button(f"Copy Bullets", key=f"copy_bullets_{job_id}"):
                            bullet_text = "\n".join(
                                f"{j}. {b}"for j, b in enumerate(bullets, 1)
                            )
                            st.code(bullet_text)

                    # Cover letter
                    if cover_letters and job_id in cover_letters:
                        st.subheader("Cover Letter")
                        letter = cover_letters[job_id]
                        st.text_area(
                            "Cover Letter",
                            value=letter,
                            height=300,
                            key=f"cover_letter_{job_id}",
                            label_visibility="collapsed",
                        )
                        if st.button(f"Copy Letter", key=f"copy_letter_{job_id}"):
                            st.code(letter)

# ── Tab 4: Export ────────────────────────────────────────────────────────────

with tab_export:
    if not st.session_state.get("has_run"):
        st.info("Run a search first to export results.")
    else:
        result = st.session_state.get("pipeline_result")
        jobs_list = result.get("jobs", []) if result else []

        if jobs_list:
            st.subheader("Download Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                json_data = jobs_to_json(jobs_list)
                st.download_button(
                    label="Download Jobs as JSON",
                    data=json_data,
                    file_name="jobs.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with col2:
                csv_data = jobs_to_csv(jobs_list)
                st.download_button(
                    label="Download Jobs as CSV",
                    data=csv_data,
                    file_name="jobs.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col3:
                cover_letters = result.get("cover_letters") if result else None
                if cover_letters:
                    letters_text = "\n\n---\n\n".join(
                        f"Job: {job_id}\n\n{letter}"
                        for job_id, letter in cover_letters.items()
                    )
                    st.download_button(
                        label="Download Cover Letters",
                        data=letters_text,
                        file_name="cover_letters.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "Cover Letters (v2)",
                        disabled=True,
                        use_container_width=True,
                        help="Cover letter export will be available in v2",
                    )
        else:
            st.warning("No jobs to export. Run a search first.")
