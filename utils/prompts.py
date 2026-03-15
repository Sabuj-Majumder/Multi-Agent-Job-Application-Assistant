"""LLM prompt templates for all agents.

All prompt strings used by agents live exclusively in this module.
No agent file should contain inline prompt text.
"""

# ── Agent 2: Resume Analyzer ────────────────────────────────────────────────

RESUME_ANALYZER_PROMPT: str = """
You are an expert technical recruiter. Extract structured information from the resume below.

Return ONLY a valid JSON object with these exact keys:
{{
  "name": "string or null",
  "email": "string or null",
  "skills": ["list", "of", "technical", "skills"],
  "experience_years": integer or null,
  "job_titles": ["list of past job titles"],
  "education": ["list of education entries"],
  "summary": "2-3 sentence professional summary of this candidate"
}}

Rules:
- skills: include programming languages, frameworks, tools, cloud platforms, methodologies
- experience_years: estimate total years of professional experience
- job_titles: only include actual job titles held, not descriptions
- summary: write in third person, focus on technical strengths
- Return ONLY the JSON, no explanation, no markdown fences

RESUME:
{resume_text}
"""

RESUME_ANALYZER_RETRY_PROMPT: str = """
Your previous response was not valid JSON. Please try again.

Return ONLY a valid JSON object — no markdown fences, no explanation, no extra text.
The JSON must have these exact keys: name, email, skills, experience_years, job_titles, education, summary.

RESUME:
{resume_text}
"""

# ── Agent 3: Fit Scorer (v2) ────────────────────────────────────────────────

FIT_SCORER_PROMPT: str = """
You are a job-fit scoring expert. Score how well this candidate matches the job.

JOB:
- Title: {job_title}
- Company: {company}
- Description (first 1500 chars): {description}

CANDIDATE:
- Skills: {skills}
- Experience: {experience_years} years
- Past Titles: {job_titles}

Return ONLY a valid JSON object:
{{
  "score": <integer 0-100>,
  "reasoning": "<1-2 sentence explanation>"
}}

Score rubric:
- 90-100: Strong match — title aligns, 80%+ skill overlap, experience level fits
- 70-89: Good match — most skills present, minor gaps
- 50-69: Partial match — some relevant skills, significant gaps
- Below 50: Poor fit — major mismatches in skills or experience level

Return ONLY the JSON, no explanation, no markdown fences.
"""

# ── Agent 4: Resume Tailor (v2) ──────────────────────────────────────────────

RESUME_TAILOR_PROMPT: str = """
You are an expert technical resume writer. Rewrite the candidate's experience as 5 bullet points tailored to this specific job.

JOB:
- Title: {job_title}
- Company: {company}
- Description (first 2000 chars): {job_description}

CANDIDATE RESUME TEXT:
{resume_text}

Instructions:
- Rewrite the candidate's experience into 5 bullet points tailored to this specific job.
- Use keywords and technologies mentioned in the job description where truthfully applicable.
- Every bullet must begin with a strong action verb (e.g., Built, Designed, Optimized) and include a measurable outcome where possible.
- STRICTLY DO NOT invent experience, tools, or achievements not present in the original resume. Base everything strictly on the candidate's actual resume.
- Each bullet should be a single string.
- Return ONLY a valid JSON array of 5 strings — no preamble, no explanation, no markdown fences.

Example output:
[
  "Engineered a scalable data pipeline using Python and AWS, reducing data processing time by 40%.",
  "Designed RESTful APIs using Node.js and Express, enabling support for 10M+ daily requests."
]
"""


# ── Agent 5: Cover Letter (v2) ───────────────────────────────────────────────

COVER_LETTER_PROMPT: str = """
You are a professional cover letter writer. Write a personalized cover letter for this specific job.

JOB:
- Title: {job_title}
- Company: {company}
- Description: {description}

CANDIDATE:
- Name: {candidate_name}
- Skills: {skills}
- Experience: {experience_years} years
- Past Titles: {past_titles}
- Summary: {summary}

TAILORED BULLETS (if available):
{tailored_bullets}

Instructions:
- Write exactly 3 paragraphs:
  1. Opening: Why this role excites the candidate (reference specific company/role details)
  2. Body: Specific relevant experience and skills that match the job requirements
  3. Closing: Call to action, enthusiasm for next steps
- Keep it ~250 words
- Professional but not generic — must reference specific job details
- Write in first person

Return ONLY the cover letter text, no additional commentary.
"""
