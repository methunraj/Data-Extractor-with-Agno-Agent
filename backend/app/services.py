# app/services.py
import os
import json
import uuid
import time
import glob
import logging
import pandas as pd
from pathlib import Path
import asyncio
import traceback
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional

# Set Matplotlib backend to a non-GUI backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib imports

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.messages import HumanMessage
from .core.config import settings

logger = logging.getLogger(__name__)

# Agent pool for reusing initialized agents
AGENT_POOL: Dict[str, Any] = {}

@lru_cache(maxsize=10)
def create_langchain_agent(model: str, temp_dir: str) -> Any:
    """Creates a fully autonomous LangChain agent that can handle ANY data format."""
    
    # Check if agent exists in pool
    agent_key = f"{model}_{temp_dir}"
    if agent_key in AGENT_POOL:
        logger.info(f"Reusing cached agent for model: {model}")
        return AGENT_POOL[agent_key]
    
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.7,
        verbose=True,
        thinking_budget=128
    )

    # AUTONOMOUS AGENT INSTRUCTIONS
    search_instructions = "- Google Search: Get current exchange rates and other real-time data" if settings.GOOGLE_CSE_ID else "- Note: Google Search is not available - use static exchange rates or online APIs via requests"
    
    autonomous_instructions = f"""
                                🎯 **AUTONOMOUS DATA VIRTUOSO & EXCEL ARCHITECT**

                                You are an expert financial analyst. Your goal is to create a professional, multi-sheet Excel report from JSON data.

                                **Core Task: Generate and Execute a Python Script**
                                1. Your primary output is a single Python script that performs the entire data transformation and Excel generation process.
                                2. **Crucially, when you create your script, save it as 'excel_report_generator.py' in the temp directory.** This is a mandatory step.
                                3. **IMPORTANT:** When using matplotlib, you MUST set the backend to 'Agg' at the beginning of your script with `matplotlib.use('Agg')` before any other matplotlib imports to avoid threading issues.

                                **Script Requirements:**
                                1. **Currency Conversion:** The script must detect the original currency, use the Google Search tool to find the current USD exchange rate, and then create two columns for every financial figure: one for the original currency and an adjacent one for the converted USD value. Add a note in the summary sheet citing the rate used.
                                2. **Data Integrity:** The script must not invent or alter historical data. Forecasts are allowed but must be clearly labeled and based on the source data.
                                3. **Excel Structure:** The script should create an Excel file with multiple sheets:
                                - A 'Summary' sheet with a narrative, key insights, and an embedded forecast chart (in USD).
                                - Separate sheets for 'Income Statement', 'Balance Sheet', and 'Cash Flow'.
                                - Additional sheets for any data breakdowns found, like 'Business Segments' or 'Geographic Revenue'.
                                4. **No Raw JSON:** The final Excel file must NOT contain any raw JSON data.

                                **Additional Autonomous Capabilities:**
                                • **Universal Data Parser**: Write Python code to parse ANY input format, including malformed JSON
                                • **Self-Healing**: Automatically detect and fix data parsing errors
                                • **Creative Problem Solver**: Find alternative approaches when standard methods fail
                                • **Professional Formatting**: Apply sophisticated styling with colors, borders, and fonts

                                **Autonomous Workflow:**
                                1. **ANALYZE INPUT**: Write Python code to examine the input data structure
                                2. **PARSE ROBUSTLY**: Handle ANY format - JSON, malformed JSON, CSV, text, etc.
                                3. **DETECT FINANCIAL DATA**: Identify currency fields, financial statements, and time series
                                4. **CURRENCY CONVERSION**: Use Google Search to get current exchange rates for conversion to USD
                                5. **CREATE STRUCTURED EXCEL**: 
                                - Summary sheet with narrative and insights
                                - Financial statement sheets (Income Statement, Balance Sheet, Cash Flow)
                                - Business segment breakdowns if applicable
                                - Charts and visualizations (especially forecast charts in USD)
                                6. **VALIDATE OUTPUT**: Ensure the Excel file is created with all required sheets

                                **Your Tools:**
                                - PythonREPLTool: Write and execute ALL Python code autonomously
                                - Google Search: Get current exchange rates and other real-time data


                                **YOUR TOOLS:**
                                - PythonREPLTool: Write and execute ALL Python code autonomously
                                {search_instructions}

                                Remember: You are FULLY AUTONOMOUS. Parse the data yourself, handle all errors, and create exceptional Excel files!
"""
    
    # Create tools
    python_tool = PythonREPLTool()
    
    # Create Google Search tool - only if CSE ID is configured
    tools = [python_tool]
    
    if settings.GOOGLE_CSE_ID:
        search = GoogleSearchAPIWrapper(
            google_api_key=settings.GOOGLE_API_KEY,
            google_cse_id=settings.GOOGLE_CSE_ID
        )
        search_tool = Tool(
            name="google_search",
            description="Search Google for recent results, especially useful for currency exchange rates.",
            func=search.run,
        )
        tools.append(search_tool)
    else:
        logger.warning("GOOGLE_CSE_ID not configured. Google Search functionality will be disabled.")
    

    
    # Create the autonomous agent using LangChain
    from langchain import hub
    from langchain_core.prompts import PromptTemplate
    
    # Use a custom prompt template that includes our autonomous instructions
    template = autonomous_instructions + """

Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    
    # Create the autonomous agent using LangChain
    agent = create_react_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    logger.info(f"Created autonomous LangChain agent with model: {model}")
    
    # Store agent executor in pool for reuse
    AGENT_POOL[agent_key] = agent_executor
    return agent_executor

async def direct_json_to_excel_async(json_data: str, file_name: str, chunk_size: int, temp_dir: str) -> Tuple[str, str, str]:
    """Simple fallback that just saves raw data to Excel - agent should handle everything autonomously."""
    return await asyncio.to_thread(direct_json_to_excel, json_data, file_name, chunk_size, temp_dir)

def direct_json_to_excel(json_data: str, file_name: str, chunk_size: int, temp_dir: str) -> Tuple[str, str, str]:
    """
    Simple fallback conversion - just save raw data to Excel.
    The autonomous agent should handle all complex parsing.
    """
    try:
        file_id = str(uuid.uuid4())
        safe_filename = "".join(c for c in file_name if c.isalnum() or c in (' ', '-', '_')).strip()
        xlsx_filename = f"{safe_filename}_direct.xlsx"
        file_path = os.path.join(temp_dir, f"{file_id}_{xlsx_filename}")

        # Simple approach - let pandas handle what it can
        try:
            # Try to parse as JSON
            data = json.loads(json_data)
            df = pd.json_normalize(data) if isinstance(data, (dict, list)) else pd.DataFrame([{'data': str(data)}])
        except:
            # If JSON parsing fails, just put the raw data in Excel
            df = pd.DataFrame([{'raw_data': json_data}])

        # Save to Excel
        df.to_excel(file_path, index=False)
        return file_id, xlsx_filename, file_path
            
    except Exception as e:
        logger.error(f"Direct conversion failed: {e}")
        raise

async def convert_with_langchain_async(json_data: str, file_name: str, description: str, model: str, temp_dir: str) -> str:
    """Async version for autonomous agent processing."""
    return await asyncio.to_thread(convert_with_langchain, json_data, file_name, description, model, temp_dir)

def convert_with_langchain(json_data: str, file_name: str, description: str, model: str, temp_dir: str) -> str:
    """
    Let the autonomous agent handle EVERYTHING - parsing, error fixing, Excel creation.
    """
    max_retries = 3
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            agent = create_langchain_agent(model, temp_dir)
            
            # Additional guidance for retries
            retry_guidance = ""
            if retry_count > 0:
                retry_guidance = f"""
IMPORTANT: Previous attempt failed with: {last_error}
Please try a different approach to parse and process the data.
"""
            
            # Give the raw data to the agent - let it handle everything autonomously
            prompt = f"""
                        Please process the following data and create a professional, multi-sheet Excel report.

                        Target filename: {file_name}
                        Target directory: {temp_dir}
                        Description: {description}

                        {retry_guidance}

                        **RAW INPUT DATA:**
                        {json_data}

                        **CRITICAL REQUIREMENTS FOR YOUR PYTHON SCRIPT:**

                        1. **Script Management**: 
                        - Create a Python script named 'excel_report_generator.py'
                        - Save it to: {temp_dir}/excel_report_generator.py

                        2. **Data Parsing**:
                        - Parse the input data robustly (handle ANY format including malformed JSON)
                        - Implement try-except blocks with fallback strategies
                        - Never fail due to data format issues

                        3. **Currency Conversion** (if financial data detected):
                        - Detect the original currency in the data
                        - Use Google Search to find current USD exchange rate
                        - Create TWO columns for every financial figure:
                            * Original currency value
                            * USD converted value (adjacent column)
                        - Add a note in the Summary sheet citing the exchange rate used

                        4. **Excel Structure Requirements**:
                        - **Summary Sheet**: Professional narrative with key insights and embedded forecast chart (in USD)
                        - **Income Statement**: If financial data is present
                        - **Balance Sheet**: If financial data is present  
                        - **Cash Flow**: If financial data is present
                        - **Business Segments**: If segment data is found
                        - **Geographic Revenue**: If geographic data is found
                        - **NO RAW JSON**: Do not include raw JSON data in any sheet

                        5. **Data Integrity**:
                        - Do NOT invent or alter historical data
                        - Forecasts are allowed but must be clearly labeled as projections
                        - Base all analysis on actual source data

                        6. **Professional Output**:
                        - Apply professional formatting (colors, borders, fonts)
                        - Create meaningful charts and visualizations
                        - Ensure file saves to: {temp_dir}/{file_name}.xlsx

                        Remember: You are an expert financial analyst. Create an exceptional, professional Excel report!
                        
                        """
            
            # Execute the autonomous agent
            result = agent.invoke({"input": prompt})
            
            # Extract response
            response_content = "Processing completed"
            if result and "output" in result:
                response_content = str(result["output"])
            elif result and isinstance(result, dict):
                # Fallback to searching for any text content
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 10:
                        response_content = str(value)
                        break
            
            return response_content
            
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            error_details = traceback.format_exc()
            logger.error(f"Autonomous agent processing failed (attempt {retry_count}/{max_retries}): {e}\n{error_details}")
            
            if retry_count >= max_retries:
                logger.error(f"All {max_retries} attempts failed.")
                raise
            
            time.sleep(retry_count * 2)
            logger.info(f"Retrying with autonomous agent (attempt {retry_count+1}/{max_retries})...")

def find_newest_file(directory: str, files_before: set) -> Optional[str]:
    """Find the newest Excel file created after the given set of files."""
    files_after = set(glob.glob(os.path.join(directory, "*.xlsx")))
    new_files = files_after - files_before
    if not new_files:
        return None
    return max(new_files, key=os.path.getmtime)

def cleanup_agent_pool():
    """Remove agents from the pool to free up memory."""
    global AGENT_POOL
    logger.info(f"Cleaning up agent pool with {len(AGENT_POOL)} agents")
    AGENT_POOL.clear()