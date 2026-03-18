# LangSmith Observability

What LangSmith tracing gives you:
- Trace every agent call in the LangGraph pipeline
- See inputs/outputs for all agents
- Monitor latency per node and token usage. 

![LangSmith Trace](./langsmith_trace.png)
*Add your own screenshot here after first run*

### How to Find Your Traces

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Select the project `job-assistant`
3. Click any run to see the full agent graph execution

> **Note:** LangSmith tracing is entirely optional. It is free up to 5,000 traces/month on LangSmith's free tier. 

If `LANGCHAIN_API_KEY` is missing in your `.env` file, the app runs exactly as before with zero errors. Tracing failures are silent.
