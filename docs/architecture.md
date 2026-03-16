# Architecture — Multi-Agent Job Application Assistant

## System Overview

This project implements a **multi-agent AI pipeline** using LangGraph that autonomously scrapes job listings, analyzes resumes, scores job-candidate fit, tailors resume content, and generates cover letters.

## Agent Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐   │
│  │  Search   │  │ Results  │  │   Tailored   │  │  Export   │   │
│  │   Tab     │  │   Tab    │  │  Materials   │  │   Tab     │   │
│  └──────────┘  └──────────┘  └──────────────┘  └──────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ pipeline.invoke(state)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LangGraph Pipeline (graph.py)                  │
│                                                                  │
│  START ──→ job_scraper ──→ [conditional] ──→ resume_analyzer    │
│                                │                    │            │
│                                │ (no resume)        │            │
│                                ▼                    ▼            │
│                               END                  END           │
│                                                                  │
│  v2 additions:                                                   │
│  ... ──→ fit_scorer ──→ resume_tailor ──→ cover_letter ──→ END  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Shared State (AgentState)                      │
│  ┌───────────┐  ┌─────────────────┐  ┌───────────────────────┐ │
│  │ User Input │  │  Agent Outputs   │  │ Pipeline Metadata     │ │
│  │ job_title  │  │  jobs            │  │ error                 │ │
│  │ location   │  │  candidate_prof  │  │ active_agent          │ │
│  │ resume_txt │  │  ranked_jobs     │  │ completed_agents      │ │
│  └───────────┘  └─────────────────┘  └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

## External APIs

| API       | Purpose                    | Auth Required |
|-----------|----------------------------|---------------|
| The Muse  | Job aggregation            | Yes (free)    |
| Arbeitnow | Job aggregation            | Yes (free)    |
| Groq      | LLM inference (LLaMA 3.3) | Yes (free)    |

## Tech Stack

- **Orchestration:** LangGraph (explicit state graph)
- **LLM:** Groq (LLaMA 3.3 70B for quality, LLaMA 3.1 8B for speed)
- **Data Models:** Pydantic v2
- **UI:** Streamlit
- **Logging:** structlog (JSON)
- **Testing:** pytest with mocked HTTP
```
