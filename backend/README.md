# Enhanced LangChain AI JSON to Excel API

A high-performance API for converting JSON data to Excel files using LangChain and Google Gemini AI capabilities.

## Features

This API provides autonomous JSON to Excel conversion with advanced AI capabilities:

### Core Features

- **Autonomous Data Processing**: Handles ANY input format including malformed JSON
- **Self-Healing Parser**: Automatically detects and fixes data parsing errors
- **Professional Excel Output**: Creates multi-sheet workbooks with charts and formatting
- **Currency Conversion**: Converts financial data to USD (with Google Search integration)
- **Agent Pooling**: Reuses initialized agents to avoid recreation costs
- **Retry Logic**: Automatically retries failed operations with exponential backoff

### LangChain Integration

- **ReAct Agent**: Uses LangChain's ReAct pattern for autonomous reasoning
- **Tool Integration**: Python REPL and Google Search tools
- **Google Gemini**: Powered by Google's latest Gemini 2.5 Flash model
- **Graceful Fallback**: Falls back to direct conversion if AI processing fails

### API Modes

- **Auto Mode**: Tries AI processing first, falls back to direct conversion
- **AI Only**: Forces AI processing (fails if AI cannot process)
- **Direct Only**: Bypasses AI for simple data structures

## Recent Fixes

Fixed critical issues in the LangChain implementation:

1. **Google Search Configuration**: Made Google Custom Search Engine ID optional
2. **Graceful Degradation**: AI agents work without Google Search when CSE ID is not configured
3. **Metrics Endpoint**: Fixed duplicate parameter error in SystemMetrics
4. **Error Handling**: Improved error handling for missing environment variables

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn app.main:app --reload
```

API endpoints:

- `POST /process` - Process JSON data to Excel
- `GET /download/{file_id}` - Download generated Excel file
- `GET /metrics` - Get system metrics

## Configuration

### Required Environment Variables

- `GOOGLE_API_KEY` - Required for Google Gemini model access

### Optional Environment Variables

- `GOOGLE_CSE_ID` - Google Custom Search Engine ID for advanced search features
  - Get this from: https://cse.google.com/
  - If not provided, Google Search will be disabled but the API will still work
- `LANGCHAIN_TRACING_V2` - Enable LangChain tracing (default: false)
- `LANGCHAIN_API_KEY` - LangSmith API key for tracing
- `LOG_LEVEL` - Logging level (default: INFO)

### Example .env file

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional
GOOGLE_CSE_ID=your_google_cse_id_here
LANGCHAIN_TRACING_V2=false
LOG_LEVEL=INFO
```

## Dependencies

- fastapi
- uvicorn
- pandas
- langchain
- langchain-google-genai
- langchain-google-search
- langgraph
- openpyxl
- matplotlib 