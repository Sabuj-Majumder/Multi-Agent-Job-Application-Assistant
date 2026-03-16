"""Streamlit UI for the Multi-Agent Job Application Assistant.

Run with: streamlit run app.py
"""

import csv
import io
import json
from datetime import datetime, timedelta, timezone
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
if "job_tracker" not in st.session_state:
    st.session_state["job_tracker"] = {}


# ── Helper Functions ─────────────────────────────────────────────────────────


def filter_by_date(jobs, date_filter):
    """
    Filters jobs by how recently they were posted.
    Jobs without a date field are always included when filter is 'Any time',
    but excluded for all other filter values since we cannot verify recency.
    """
    if date_filter == "Any time":
        return jobs

    cutoff_map = {
        "Past 24 hours": timedelta(days=1),
        "Past 3 days":   timedelta(days=3),
        "Past week":     timedelta(days=7),
        "Past 2 weeks":  timedelta(days=14),
        "Past month":    timedelta(days=30),
    }
    cutoff = datetime.now(timezone.utc) - cutoff_map[date_filter]
    filtered = []
    for job in jobs:
        posted_at = getattr(job, "posted_at", None)
        if posted_at is None:
            # Fallback for dictionaries since jobs can be dicts or Job objects here
            if isinstance(job, dict):
                posted_at = job.get("posted_at")
            if posted_at is None:
                continue   # exclude undated jobs when a filter is active
        try:
            if isinstance(posted_at, str):
                # Handle ISO format strings with or without timezone
                posted_dt = datetime.fromisoformat(
                    posted_at.replace("Z", "+00:00")
                )
            else:
                posted_dt = posted_at
            # Make naive datetimes timezone-aware (assume UTC)
            if posted_dt.tzinfo is None:
                posted_dt = posted_dt.replace(tzinfo=timezone.utc)
            if posted_dt >= cutoff:
                filtered.append(job)
        except (ValueError, TypeError):
            continue   # skip jobs with unparseable dates
    return filtered


def human_readable_date(posted_at_str):
    """Convert ISO date string to friendly relative label."""
    if not posted_at_str:
        return "Date unknown"
    try:
        posted_dt = datetime.fromisoformat(posted_at_str.replace("Z", "+00:00"))
        if posted_dt.tzinfo is None:
            posted_dt = posted_dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - posted_dt
        if delta.days == 0:
            return "Posted today"
        elif delta.days == 1:
            return "Posted yesterday"
        elif delta.days <= 7:
            return f"Posted {delta.days} days ago"
        elif delta.days <= 30:
            weeks = delta.days // 7
            return f"Posted {weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            months = delta.days // 30
            return f"Posted {months} month{'s' if months > 1 else ''} ago"
    except (ValueError, TypeError):
        return "Date unknown"


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


def get_score_emoji(score: int) -> str:
    """Get emoji indicator for a fit score value.

    Args:
        score: Fit score integer 0-100.

    Returns:
        Emoji string: \"🟢\" for ≥70, \"🟡\" for 40-69, \"🔴\" for <40.
    """
    if score >= 70:
        return "🟢"
    elif score >= 40:
        return "🟡"
    return "🔴"


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
    - The Muse (themuse.com/api) — no key needed
    - [RemoteOK](https://remoteok.com/api) — Remote jobs (no key needed)
    - Arbeitnow (arbeitnow.com/api) — no key needed
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

    tracker_state = st.session_state.get("job_tracker", {})
    if tracker_state:
        st.subheader("📌 Tracker Summary")
        saved_count = sum(1 for status in tracker_state.values() if status == "saved")
        applied_count = sum(1 for status in tracker_state.values() if status == "applied")
        rejected_count = sum(1 for status in tracker_state.values() if status == "rejected")
        st.markdown(f"⭐ Saved: {saved_count} &nbsp; ✅ Applied: {applied_count} &nbsp; ❌ Rejected: {rejected_count}")
        st.divider()

    st.caption("Built with LangGraph · Groq · Streamlit")
    st.caption("[GitHub Repository](#)")


# ── Main Area — 4 Tabs ──────────────────────────────────────────────────────

st.title("Job Application Assistant")
st.caption("Multi-agent AI system for autonomous job searching and resume analysis")

tab_search, tab_results, tab_tracker, tab_tailored, tab_export = st.tabs(
    ["Search", "Results", "📌 Tracker", "Tailored Materials (v2)", "Export"]
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
        date_posted = st.selectbox(
            "Date Posted",
            options=["Any time", "Past 24 hours", "Past 3 days", "Past week", "Past 2 weeks", "Past month"],
            index=0,
            key="date_posted_filter"
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
                st.write("Job Scraper Agent — fetching from RemoteOK, The Muse, Arbeitnow...")

                # Build initial state
                initial_state: Dict[str, Any] = {
                    "job_title": job_title,
                    "location": location,
                    "num_results": int(num_results),
                    "date_filter": date_posted,
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

                        # Fit Scorer status
                        if result.get("candidate_profile"):
                            ranked = result.get("ranked_jobs", [])
                            if ranked:
                                st.write(f"Fit Scorer complete — scored {len(ranked)} jobs")
                            else:
                                st.write("Fit Scorer — scoring jobs...")

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
                        "Minimum Fit Score",
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

            filtered_jobs = filter_by_date(filtered_jobs, st.session_state.get("date_posted_filter", "Any time"))

            if not filtered_jobs and jobs_list:
                st.warning(
                    f"No jobs found for the selected date filter: **{st.session_state.get('date_posted_filter')}**. "
                    "Try a wider date range or select 'Any time'."
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
                        
                        st.caption(f"🕐 {human_readable_date(job_dict.get('posted_at'))}")

                        url = job_dict.get("url", "")
                        if url:
                            st.link_button("View Job →", url)

                    # Fit score display
                    fit_score = job_dict.get("fit_score")
                    if fit_score is not None:
                        emoji = get_score_emoji(fit_score)
                        st.markdown(f"**Fit Score:** {emoji} {fit_score}/100")
                        st.progress(fit_score / 100)
                        reasoning = job_dict.get("fit_reasoning", "")
                        if reasoning:
                            st.caption(reasoning)

                    # Description
                    description = job_dict.get("description", "")
                    if len(description) > 400:
                        st.markdown(description[:400] + "...")
                    elif description:
                        st.markdown(description)

                    # Tracker Buttons
                    st.divider()
                    job_id = job_dict.get("id", "")
                    status = st.session_state["job_tracker"].get(job_id, "none")
                    
                    if status == "saved":
                        st.markdown("🟡 Saved")
                    elif status == "applied":
                        st.markdown("🟢 Applied")
                    elif status == "rejected":
                        st.markdown("🔴 Rejected")
                        
                    btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3])
                    with btn_col1:
                        if st.button("⭐ Save", key=f"save_{job_id}", type="primary" if status == "saved" else "secondary"):
                            st.session_state["job_tracker"][job_id] = "saved"
                            st.rerun()
                    with btn_col2:
                        if st.button("✅ Applied", key=f"applied_{job_id}", type="primary" if status == "applied" else "secondary"):
                            st.session_state["job_tracker"][job_id] = "applied"
                            st.rerun()
                    with btn_col3:
                        if st.button("❌ Reject", key=f"reject_{job_id}", type="primary" if status == "rejected" else "secondary"):
                            st.session_state["job_tracker"][job_id] = "rejected"
                            st.rerun()
        else:
            st.warning("No jobs found. Try adjusting your search criteria.")

# ── Tab 3: Tracker ───────────────────────────────────────────────────────────

with tab_tracker:
    jobs = []
    if st.session_state.get("pipeline_result"):
        jobs = st.session_state["pipeline_result"].get("jobs", [])
        
    if not jobs:
        st.info("Run the agent pipeline first to see jobs here.")
    else:
        st.subheader("Job Application Tracker")
        
        tracker_state = st.session_state["job_tracker"]
        
        # Helper to get job by id
        def get_job_by_id(jid):
            for j in jobs:
                if isinstance(j, Job):
                    if j.id == jid: return j.model_dump()
                elif isinstance(j, dict):
                    if j.get("id") == jid: return j
            return None

        saved_jobs = []
        applied_jobs = []
        rejected_jobs = []
        
        for jid, status in tracker_state.items():
            job_data = get_job_by_id(jid)
            if not job_data: continue
            if status == "saved": saved_jobs.append(job_data)
            elif status == "applied": applied_jobs.append(job_data)
            elif status == "rejected": rejected_jobs.append(job_data)
            
        col1, col2, col3 = st.columns(3)
        col1.metric("⭐ Saved", len(saved_jobs))
        col2.metric("✅ Applied", len(applied_jobs))
        col3.metric("❌ Rejected", len(rejected_jobs))
        
        st.divider()
        
        def render_job_table(jobs_list, empty_msg):
            if not jobs_list:
                st.caption(empty_msg)
                return
            
            header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([3, 2, 2, 1, 1])
            with header_col1: st.markdown("**Title**")
            with header_col2: st.markdown("**Company**")
            with header_col3: st.markdown("**Location**")
            with header_col4: st.markdown("**Fit Score**")
            with header_col5: st.markdown("**Action**")
            
            st.markdown("---")
            
            for j in jobs_list:
                c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1, 1])
                with c1: st.write(j.get("title", ""))
                with c2: st.write(j.get("company", ""))
                with c3: st.write(j.get("location", ""))
                with c4: 
                    score = j.get("fit_score")
                    st.write(f"{score}/100" if score is not None else "-")
                with c5:
                    url = j.get("url", "")
                    if url: st.link_button("View →", url)
        
        with st.expander("Saved Jobs", expanded=True):
            render_job_table(saved_jobs, "No jobs in this category yet.")
            
        with st.expander("Applied Jobs", expanded=True):
            render_job_table(applied_jobs, "No jobs in this category yet.")
            
        with st.expander("Rejected Jobs", expanded=True):
            render_job_table(rejected_jobs, "No jobs in this category yet.")
            
        st.divider()
        if st.button("🗑️ Clear All Tracking Data", type="secondary"):
            st.session_state["job_tracker"] = {}
            st.rerun()

# ── Tab 4: Tailored Materials (v2) ───────────────────────────────────────────

with tab_tailored:
    if not st.session_state.get("has_run"):
        st.info("Run a search first to see tailored materials here.")
    else:
        result = st.session_state.get("pipeline_result")

        tailored_bullets = result.get("tailored_bullets") if result else None
        cover_letters = result.get("cover_letters") if result else None

        if not tailored_bullets:
            st.info("Upload a resume and run the pipeline to generate tailored bullets.")
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

                st.subheader(f"{title} at {company}")
                
                if job_id in tailored_bullets:
                    bullets = tailored_bullets[job_id]
                    if bullets:
                        for j, bullet in enumerate(bullets, 1):
                            st.markdown(f"{j}. {bullet}")
                        
                        bullet_text = "\n".join(f"{j}. {b}" for j, b in enumerate(bullets, 1))
                        if st.button("📋 Copy All Bullets", key=f"copy_{job_id}"):
                            st.code(bullet_text)
                    else:
                        st.warning("Failed to generate tailored bullets for this job.")
                
            st.caption("⚠️ Review all bullets carefully. Never submit AI-generated content without verifying accuracy.")

# ── Tab 5: Export ────────────────────────────────────────────────────────────

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
