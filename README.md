# Multi-Agent Job Application Assistant

A **multi-agent AI system** built with LangGraph that autonomously finds job listings, analyzes resumes, scores job-candidate fit, tailors resume bullet points, and writes personalized cover letters — powered entirely by free-tier APIs.

##  Architecture

```
START → Job Scraper → [conditional] → Resume Analyzer → END
```

- **Job Scraper Agent** — Fetches from 3 free job APIs (Adzuna, RemoteOK, Jooble), normalizes and deduplicates results
- **Resume Analyzer Agent** — Extracts structured candidate profile from uploaded PDF using LLM
- **Fit Scorer Agent** (v2) — Scores each job 0–100 against candidate profile
- **Resume Tailor Agent** (v2) — Rewrites resume bullets to match top jobs
- **Cover Letter Agent** (v2) — Generates personalized cover letters per job

##  Tech Stack

| Technology | Purpose |
|-----------|---------|
| **LangGraph** | Multi-agent orchestration graph |
| **LangChain + Groq** | LLM inference (LLaMA 3.3 70B, free tier) |
| **Streamlit** | Web UI |
| **Pydantic v2** | Typed data models between agents |
| **structlog** | Structured JSON logging |
| **pypdf** | PDF resume text extraction |
| **pytest** | Unit testing with mocked HTTP |

##  Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd job-assistant
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

#### Groq (Required — free)
1. Go to https://console.groq.com
2. Sign up / log in
3. Create an API key
4. Paste into `GROQ_API_KEY` in `.env`

#### Adzuna (Optional — free, instant approval)
1. Go to https://developer.adzuna.com
2. Register for an account
3. Copy your App ID and App Key
4. Paste into `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` in `.env`

#### Jooble (Optional — free)
1. Go to https://jooble.org/api/index
2. Register for API access
3. Copy your API key
4. Paste into `JOOBLE_API_KEY` in `.env`

#### RemoteOK
No API key needed! Works immediately.

### 3. Run the app

```bash
streamlit run app.py
```

Opens at http://localhost:8501

##  Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with short tracebacks
pytest tests/ -v --tb=short

# Run a single test file
pytest tests/test_job_scraper.py -v
```

##  Project Structure

```
├── app.py                      # Streamlit UI entry point
├── graph.py                    # LangGraph pipeline definition
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── agents/
│   ├── job_scraper_agent.py    # Agent 1: 3-source job fetcher
│   ├── resume_analyzer_agent.py # Agent 2: LLM-based resume parsing
│   ├── fit_scorer_agent.py     # Agent 3: Job-fit scoring (v2 stub)
│   ├── resume_tailor_agent.py  # Agent 4: Resume bullet rewriting (v2 stub)
│   └── cover_letter_agent.py   # Agent 5: Cover letter generation (v2 stub)
├── utils/
│   ├── state.py                # Pydantic models + AgentState TypedDict
│   ├── llm.py                  # Groq LLM factory
│   ├── prompts.py              # All LLM prompt templates
│   └── logger.py               # structlog configuration
├── tests/
│   ├── test_state.py           # Data model validation tests
│   ├── test_job_scraper.py     # Scraper tests with mocked HTTP
│   ├── test_resume_analyzer.py # Resume analyzer tests
│   └── fixtures/               # Test data files
└── docs/
    └── architecture.md         # System design documentation
```

##  Deployment

### Streamlit Cloud (Recommended, Free)
1. Push repo to GitHub (`.env` is gitignored)
2. Go to https://share.streamlit.io → "New app" → select your repo
3. Set `app.py` as entry point
4. Add API keys in the "Secrets"section
5. Deploy — get a public URL instantly

### Railway (Alternative, Free Tier)
1. Connect GitHub repo to Railway
2. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Add env vars in Railway dashboard

##  Roadmap

- [x] Job Scraper Agent (3 API sources)
- [x] Resume Analyzer Agent (LLM extraction)
- [ ] Fit Scorer Agent (v2)
- [ ] Resume Tailor Agent (v2)
- [ ] Cover Letter Agent (v2)
- [ ] Batch mode with LangGraph map-reduce
- [ ] Job Application Tracker with SQLite

##  License

MIT
