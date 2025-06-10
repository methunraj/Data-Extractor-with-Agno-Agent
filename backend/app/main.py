# app/main.py
import os
import uuid
import time
import glob
import shutil
import logging
import tempfile
import threading
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool

from . import schemas, services
from .core.config import settings

# --- Application State and Logging ---
app_state = {}

logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Lifespan Management (Startup and Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # Startup
    app_state["TEMP_DIR"] = tempfile.mkdtemp(prefix=settings.TEMP_DIR_PREFIX)
    app_state["TEMP_FILES"] = {}
    app_state["METRICS"] = {
        'total_requests': 0, 'successful_conversions': 0, 'ai_conversions': 0,
        'direct_conversions': 0, 'failed_conversions': 0,
        'average_processing_time': 0.0
    }
    app_state["LOCK"] = threading.Lock()
    logger.info(f"Application startup complete. Temp directory: {app_state['TEMP_DIR']}")
    
    yield  # Application is now running
    
    # Shutdown
    logger.info("Application shutdown. Cleaning up temp directory...")
    # Clean up agent pool to free memory
    services.cleanup_agent_pool()
    shutil.rmtree(app_state["TEMP_DIR"])
    logger.info("Cleanup complete.")

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# --- Utility Functions ---
def update_metrics(processing_time: float, method: str, success: bool):
    with app_state["LOCK"]:
        metrics = app_state["METRICS"]
        metrics['total_requests'] += 1
        if success:
            metrics['successful_conversions'] += 1
            if method == 'ai':
                metrics['ai_conversions'] += 1
            else:
                metrics['direct_conversions'] += 1
            
            total_time = metrics['average_processing_time'] * (metrics['successful_conversions'] - 1)
            metrics['average_processing_time'] = (total_time + processing_time) / metrics['successful_conversions']
        else:
            metrics['failed_conversions'] += 1

# --- API Endpoints ---
@app.post("/process", response_model=schemas.ProcessResponse)
async def process_json_data(request: schemas.ProcessRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    processing_method = "direct"  # Default method
    
    files_before = set(glob.glob(os.path.join(app_state["TEMP_DIR"], "*.xlsx")))

    try:
        # --- AI Processing Logic ---
        if request.processing_mode in ["auto", "ai_only"]:
            try:
                # Use async version for better performance
                ai_response_content = await services.convert_with_agno_async(
                    request.json_data,
                    request.file_name,
                    request.description,
                    request.model,
                    app_state["TEMP_DIR"]
                )
                
                newest_file = services.find_newest_file(app_state["TEMP_DIR"], files_before)
                
                if newest_file:
                    processing_method = "ai"
                    file_id = str(uuid.uuid4())
                    original_filename = os.path.basename(newest_file)
                    
                    with app_state["LOCK"]:
                        app_state["TEMP_FILES"][file_id] = {'path': newest_file, 'filename': original_filename}

                    processing_time = time.time() - start_time
                    update_metrics(processing_time, processing_method, True)
                    
                    return schemas.ProcessResponse(
                        success=True, file_id=file_id, file_name=original_filename,
                        download_url=f"/download/{file_id}", ai_analysis=ai_response_content,
                        processing_method=processing_method, processing_time=processing_time,
                        data_size=len(request.json_data)
                    )
                
                if request.processing_mode == "ai_only":
                    raise HTTPException(status_code=500, detail="AI processing was requested, but no file was generated.")

            except Exception as e:
                logger.warning(f"AI processing failed: {e}. Falling back to direct conversion.")
                if request.processing_mode == "ai_only":
                    raise HTTPException(status_code=500, detail=f"AI-only processing failed: {e}")

        # --- Direct Conversion (Fallback or direct_only mode) ---
        logger.info("Using direct conversion...")
        # Use async version for better performance
        file_id, xlsx_filename, file_path = await services.direct_json_to_excel_async(
            request.json_data, request.file_name, request.chunk_size, app_state["TEMP_DIR"]
        )
        
        with app_state["LOCK"]:
            app_state["TEMP_FILES"][file_id] = {'path': file_path, 'filename': xlsx_filename}

        processing_time = time.time() - start_time
        update_metrics(processing_time, processing_method, True)

        return schemas.ProcessResponse(
            success=True, file_id=file_id, file_name=xlsx_filename,
            download_url=f"/download/{file_id}", processing_method=processing_method,
            processing_time=processing_time, data_size=len(request.json_data)
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        update_metrics(time.time() - start_time, processing_method, False)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    with app_state["LOCK"]:
        file_info = app_state["TEMP_FILES"].get(file_id)
    
    if not file_info or not os.path.exists(file_info['path']):
        raise HTTPException(status_code=404, detail="File not found or has expired.")
        
    return FileResponse(
        path=file_info['path'],
        filename=file_info['filename'],
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.get("/metrics", response_model=schemas.SystemMetrics)
async def get_system_metrics():
    with app_state["LOCK"]:
        metrics = app_state["METRICS"].copy()
        active_files = len(app_state["TEMP_FILES"])

    success_rate = (metrics['successful_conversions'] / max(metrics['total_requests'], 1)) * 100
    
    return schemas.SystemMetrics(
        **metrics,
        success_rate=round(success_rate, 2),
        average_processing_time=round(metrics['average_processing_time'], 2),
        active_files=active_files,
        temp_directory=app_state["TEMP_DIR"]
    )

@app.get("/")
async def root():
    return {"message": "Welcome to the Enhanced Agno AI API. See /docs for more info."}