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

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.python import PythonTools
from agno.tools.googlesearch import GoogleSearchTools
from .core.config import settings

logger = logging.getLogger(__name__)

# Agent pool for reusing initialized agents (lightweight in Agno ~3.75 KiB per agent)
AGENT_POOL: Dict[str, Agent] = {}

@lru_cache(maxsize=10)
def create_agno_agent(model: str, temp_dir: str) -> Agent:
    """Creates and configures the Agno agent with robust, clear instructions.
    
    Agno agents are extremely lightweight (~3.75 KiB) and fast to instantiate (~2μs).
    """
    # Check if agent exists in pool
    agent_key = f"{model}_{temp_dir}"
    if agent_key in AGENT_POOL:
        logger.info(f"Reusing cached agent for model: {model}")
        return AGENT_POOL[agent_key]
    
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in the environment.")

    # --- FINAL, SIMPLIFIED AND ROBUST INSTRUCTIONS ---
    instructions = [
        "You are an expert financial analyst. Your goal is to create a professional, multi-sheet Excel report from JSON data.",
        
        "**Core Task: Generate and Execute a Python Script**",
        "1.  Your primary output is a single Python script that performs the entire data transformation and Excel generation process.",
        "2.  **Crucially, when you save this script using the `save_to_file_and_run` tool, you MUST provide a `file_name` argument. Name the script 'excel_report_generator.py'.** This is a mandatory step.",
        "3.  **IMPORTANT:** If you use matplotlib, you MUST set the backend to 'Agg' at the beginning of your script with `matplotlib.use('Agg')` before any other matplotlib imports to avoid threading issues.",

        "**Script Requirements:**",
        "1.  **Currency Conversion:** The script must detect the original currency, use the `Google Search` tool to find the current USD exchange rate, and then create two columns for every financial figure: one for the original currency and an adjacent one for the converted USD value. Add a note in the summary sheet citing the rate used.",
        "2.  **Data Integrity:** The script must not invent or alter historical data. Forecasts are allowed but must be clearly labeled and based on the source data.",
        "3.  **Excel Structure:** The script should create an Excel file with multiple sheets:",
        "    - A 'Summary' sheet with a narrative, key insights, and an embedded forecast chart (in USD).",
        "    - Separate sheets for 'Income Statement', 'Balance Sheet', and 'Cash Flow'.",
        "    - Additional sheets for any data breakdowns found, like 'Business Segments' or 'Geographic Revenue'.",
        "4.  **No Raw JSON:** The final Excel file must NOT contain any raw JSON data.",

        "Review your generated Python code for correctness before calling the tool to save and run it.",
    ]
    # --- END OF FINAL INSTRUCTIONS ---
    
    # Create the agent with appropriate tools
    # According to docs, Agno agents are very lightweight (~3.75 KiB) and fast to instantiate (~2μs)
    agent = Agent(
        model=Gemini(id=model, api_key=settings.GOOGLE_API_KEY),
        tools=[
            PythonTools(run_code=True, pip_install=True, base_dir=Path(temp_dir)),
            GoogleSearchTools()
        ],
        show_tool_calls=True,
        instructions=instructions,
        exponential_backoff=True,  # Auto-retry with backoff on model errors
        retries=5,                 # Number of retries for model calls
    )
    logger.info(f"Created Agno agent with model: {model} and optimized configuration")
    
    # Store agent in pool for reuse - very memory-efficient in Agno
    AGENT_POOL[agent_key] = agent
    return agent

async def direct_json_to_excel_async(json_data: str, file_name: str, chunk_size: int, temp_dir: str) -> Tuple[str, str, str]:
    """Async version of direct_json_to_excel for better performance."""
    # Run the synchronous function in a thread pool to make it non-blocking
    return await asyncio.to_thread(direct_json_to_excel, json_data, file_name, chunk_size, temp_dir)

def direct_json_to_excel(json_data: str, file_name: str, chunk_size: int, temp_dir: str) -> Tuple[str, str, str]:
    """
    Convert JSON data directly to Excel with automatic retry mechanism.
    Will retry up to 3 times with different approaches on each retry.
    """
    max_retries = 3
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            decoder = json.JSONDecoder()
            pos = 0
            data_objects = []
            clean_json_data = json_data.strip()
            
            # Try different parsing approaches based on retry count
            if retry_count == 0:
                # First attempt: Standard parsing
                while pos < len(clean_json_data):
                    obj, end_pos = decoder.raw_decode(clean_json_data[pos:])
                    data_objects.append(obj)
                    pos = end_pos
                    while pos < len(clean_json_data) and clean_json_data[pos].isspace():
                        pos += 1
            elif retry_count == 1:
                # Second attempt: Line-by-line parsing
                for line in clean_json_data.split('\n'):
                    if line.strip():
                        try:
                            obj = json.loads(line.strip())
                            data_objects.append(obj)
                        except json.JSONDecodeError:
                            continue
            else:
                # Third attempt: Try with more lenient approach (wrap in array if needed)
                try:
                    data_objects = [json.loads(clean_json_data)]
                except json.JSONDecodeError:
                    try:
                        data_objects = [json.loads(f"[{clean_json_data}]")]
                    except:
                        # Last resort: Try to extract any valid JSON objects
                        import re
                        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                        matches = re.findall(json_pattern, clean_json_data)
                        for match in matches:
                            try:
                                obj = json.loads(match)
                                data_objects.append(obj)
                            except:
                                continue
            
            if not data_objects:
                raise ValueError("No valid JSON objects found in the input data")
            
            data = data_objects[0] if len(data_objects) == 1 else data_objects

            file_id = str(uuid.uuid4())
            safe_filename = "".join(c for c in file_name if c.isalnum() or c in (' ', '-', '_')).strip()
            xlsx_filename = f"{safe_filename}_direct.xlsx"
            file_path = os.path.join(temp_dir, f"{file_id}_{xlsx_filename}")

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                if isinstance(data, list):
                    if len(data) > chunk_size:
                        for i in range(0, len(data), chunk_size):
                            df = pd.json_normalize(data[i:i + chunk_size])
                            df.to_excel(writer, sheet_name=f'Data_Chunk_{i//chunk_size + 1}', index=False)
                    else:
                        pd.json_normalize(data).to_excel(writer, sheet_name='Data', index=False)
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            pd.json_normalize(value).to_excel(writer, sheet_name=str(key)[:31], index=False)
                else:
                    pd.DataFrame([{'value': data}]).to_excel(writer, sheet_name='Data', index=False)

            return file_id, xlsx_filename, file_path
            
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            error_details = traceback.format_exc()
            logger.error(f"Direct conversion failed (attempt {retry_count}/{max_retries}): {e}\n{error_details}")
            
            if retry_count >= max_retries:
                logger.error(f"All {max_retries} direct conversion attempts failed. Giving up.")
                raise
            
            # Wait briefly before retrying (with increasing delay)
            time.sleep(retry_count)
            logger.info(f"Retrying direct conversion (attempt {retry_count+1}/{max_retries})...")

async def convert_with_agno_async(json_data: str, file_name: str, description: str, model: str, temp_dir: str) -> str:
    """Async version of convert_with_agno for better performance."""
    # Run the synchronous function in a thread pool to make it non-blocking
    return await asyncio.to_thread(convert_with_agno, json_data, file_name, description, model, temp_dir)

def convert_with_agno(json_data: str, file_name: str, description: str, model: str, temp_dir: str) -> str:
    """
    Convert JSON data to Excel using Agno AI with automatic retry mechanism.
    Will retry up to 3 times with different error handling approaches on each retry.
    """
    max_retries = 3
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            agent = create_agno_agent(model, temp_dir)
            
            # Add more specific instructions on each retry
            matplotlib_instruction = """
            IMPORTANT: When using matplotlib, be sure to set the backend to 'Agg' at the top of your script:
            ```python
            import matplotlib
            matplotlib.use('Agg')  # Must be before any matplotlib imports
            ```
            """
            
            error_handling_instruction = ""
            if retry_count > 0:
                error_handling_instruction = f"""
                IMPORTANT: Previous attempt failed with error: {str(last_error)}
                
                On this retry #{retry_count+1}, please:
                1. Use try/except blocks around ALL matplotlib and file operations
                2. Add proper error handling and logging
                3. Ensure all file paths are absolute and exist before writing
                4. If creating charts, ensure to close all figure objects with plt.close()
                """
            
            prompt = f"""
            Please process the following JSON data from an annual report according to your specialized instructions.
            The final Excel file should be named based on this base name: {file_name}
            Description of the report: {description}
            
            {matplotlib_instruction}
            {error_handling_instruction}
            
            JSON Data:
            {json_data}
            """
            
            # Using run method as recommended in the documentation
            response = agent.run(prompt)
            return response.content
            
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            error_details = traceback.format_exc()
            logger.error(f"Agno AI processing failed (attempt {retry_count}/{max_retries}): {e}\n{error_details}")
            
            if retry_count >= max_retries:
                logger.error(f"All {max_retries} attempts failed. Giving up.")
                raise
            
            # Wait briefly before retrying (with increasing delay)
            time.sleep(retry_count * 2)
            logger.info(f"Retrying conversion (attempt {retry_count+1}/{max_retries})...")

def find_newest_file(directory: str, files_before: set) -> Optional[str]:
    files_after = set(glob.glob(os.path.join(directory, "*.xlsx")))
    new_files = files_after - files_before
    if not new_files:
        return None
    return max(new_files, key=os.path.getmtime)

# Cleanup function to manage the agent pool
def cleanup_agent_pool():
    """Remove agents from the pool to free up memory."""
    global AGENT_POOL
    logger.info(f"Cleaning up agent pool with {len(AGENT_POOL)} agents")
    AGENT_POOL.clear()