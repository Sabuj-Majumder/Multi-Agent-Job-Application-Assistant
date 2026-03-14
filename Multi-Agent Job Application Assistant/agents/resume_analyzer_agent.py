"""Resume Analyzer Agent — extracts structured profile from uploaded resume PDF.

LangGraph node name: "resume_analyzer"
Reads from state: resume_text
Writes to state: candidate_profile, active_agent, completed_agents
Uses LLM: llama-3.3-70b-versatile on Groq
"""

import json
import re
import time
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

from utils.llm import get_primary_llm
from utils.logger import log
from utils.prompts import RESUME_ANALYZER_PROMPT, RESUME_ANALYZER_RETRY_PROMPT
from utils.state import AgentState, CandidateProfile

# Maximum retries for LLM JSON parsing
MAX_RETRIES: int = 2


def clean_json_response(text: str) -> str:
    """Strip markdown fences and extra whitespace from LLM JSON output.

    Args:
        text: Raw LLM response string that may contain markdown code fences.

    Returns:
        Cleaned string ready for JSON parsing.
    """
    # Remove markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    return cleaned.strip()


def parse_llm_profile(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response into a profile dictionary.

    Args:
        response_text: Raw text from LLM response.

    Returns:
        Parsed dictionary if valid JSON, None otherwise.
    """
    try:
        cleaned = clean_json_response(response_text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def extract_profile_from_text(resume_text: str) -> CandidateProfile:
    """Use LLM to extract a structured CandidateProfile from resume text.

    Calls the primary Groq model with the resume analyzer prompt.
    On JSON parse failure, retries once with a stricter prompt.
    On second failure, returns a minimal CandidateProfile with raw_text.

    Args:
        resume_text: Full text content extracted from the resume PDF.

    Returns:
        A CandidateProfile with extracted fields populated.
    """
    llm = get_primary_llm(temperature=0.1)

    prompts_to_try = [RESUME_ANALYZER_PROMPT, RESUME_ANALYZER_RETRY_PROMPT]

    for attempt in range(MAX_RETRIES):
        try:
            prompt_template = ChatPromptTemplate.from_template(
                prompts_to_try[min(attempt, len(prompts_to_try) - 1)]
            )
            chain = prompt_template | llm
            response = chain.invoke({"resume_text": resume_text})

            parsed = parse_llm_profile(response.content)
            if parsed is not None:
                log.info(
                    "resume_parse_success",
                    attempt=attempt + 1,
                    skills_count=len(parsed.get("skills", [])),
                )
                return CandidateProfile(
                    name=parsed.get("name"),
                    email=parsed.get("email"),
                    skills=parsed.get("skills", []),
                    experience_years=parsed.get("experience_years"),
                    job_titles=parsed.get("job_titles", []),
                    education=parsed.get("education", []),
                    summary=parsed.get("summary", ""),
                    raw_text=resume_text,
                )

            log.warning(
                "llm_parse_failed",
                attempt=attempt + 1,
                error="Invalid JSON in LLM response",
            )

        except (json.JSONDecodeError, KeyError) as e:
            log.warning(
                "llm_parse_failed",
                attempt=attempt + 1,
                error=str(e),
            )
            continue
        except Exception as e:
            log.error(
                "llm_call_failed",
                attempt=attempt + 1,
                error=str(e),
            )
            break

    # Final fallback — return minimal profile with raw text
    log.error("resume_extraction_failed", reason="All LLM parse attempts failed")
    return CandidateProfile(raw_text=resume_text)


def resume_analyzer_agent(state: AgentState) -> AgentState:
    """LangGraph node: extracts structured candidate profile from resume text.

    Reads resume_text from state. If absent or empty, skips gracefully.
    Uses the primary Groq LLM to parse the resume into a CandidateProfile.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with candidate_profile populated.
    """
    start_time = time.time()
    resume_text: Optional[str] = state.get("resume_text")

    log.info("agent_started", agent="resume_analyzer")

    if not resume_text:
        log.warning("resume_analyzer_skipped", reason="No resume text provided")
        return {
            **state,
            "candidate_profile": None,
            "active_agent": None,
            "completed_agents": state.get("completed_agents", []) + ["resume_analyzer"],
        }

    try:
        # Log metadata only — never log raw resume content
        word_count = len(resume_text.split())
        log.info("resume_processing", word_count=word_count)

        profile = extract_profile_from_text(resume_text)

        elapsed_ms = int((time.time() - start_time) * 1000)
        log.info(
            "agent_completed",
            agent="resume_analyzer",
            skills_found=len(profile.skills),
            duration_ms=elapsed_ms,
        )

        return {
            **state,
            "candidate_profile": profile,
            "active_agent": None,
            "completed_agents": state.get("completed_agents", []) + ["resume_analyzer"],
        }

    except Exception as e:
        log.error("agent_failed", agent="resume_analyzer", error=str(e))
        return {
            **state,
            "candidate_profile": CandidateProfile(raw_text=resume_text or ""),
            "error": f"Resume analysis failed: {str(e)}",
            "active_agent": None,
            "completed_agents": state.get("completed_agents", []) + ["resume_analyzer"],
        }
