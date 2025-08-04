from __future__ import annotations

import subprocess
import os
from pathlib import Path
from typing import Optional, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel

from phosphobot.models import StatusResponse

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # dotenv not installed, that's okay

router = APIRouter(tags=["kinematics"])

# Global variable to track the AI-kinematics subprocess
kinematics_process: Optional[subprocess.Popen[str]] = None

# Global variable to store API key in memory
_anthropic_api_key: Optional[str] = None

class AITaskRequest(BaseModel):
    task: str

class ApiKeyRequest(BaseModel):
    api_key: str

@router.post("/kinematics/set-api-key", response_model=StatusResponse)
async def set_api_key(request: ApiKeyRequest) -> StatusResponse:
    """Set the Anthropic API key for AI vision features."""
    global _anthropic_api_key
    
    if not request.api_key or not request.api_key.strip():
        raise HTTPException(status_code=400, detail="API key cannot be empty")
    
    # Basic validation for Anthropic API key format
    api_key = request.api_key.strip()
    if not api_key.startswith('sk-ant-'):
        raise HTTPException(status_code=400, detail="Invalid Anthropic API key format")
    
    _anthropic_api_key = api_key
    logger.info("Anthropic API key has been set for AI-kinematics")
    
    return StatusResponse(status="ok", message="API key set successfully")

@router.get("/kinematics/api-key-status", response_model=dict)
async def get_api_key_status() -> dict:
    """Check if API key is set."""
    global _anthropic_api_key
    return {"api_key_set": _anthropic_api_key is not None and len(_anthropic_api_key) > 0}

@router.post("/kinematics/launch", response_model=StatusResponse)
async def launch_kinematics(
    background_tasks: BackgroundTasks,
    request: Request,
    mode: Literal["manual", "ai"] = Query("manual", description="Operating mode for the kinematics app"),
) -> StatusResponse:
    """Launch the AI-kinematics stream processing in *manual* or *AI* mode.

    The underlying script is *phosphobot/ai_kinematics/main.py* which now accepts
    command line arguments for non-interactive operation.
    """

    global kinematics_process, _anthropic_api_key

    # Prevent double-launch
    if kinematics_process is not None and kinematics_process.poll() is None:
        raise HTTPException(status_code=400, detail="Kinematics process already running")

    # Clean up any finished process handle
    if kinematics_process is not None:
        kinematics_process = None

    # Check API key for AI mode
    if mode == "ai" and (_anthropic_api_key is None or not _anthropic_api_key.strip()):
        raise HTTPException(status_code=400, detail="Anthropic API key is required for AI mode. Please set it first.")

    # Get task description for AI mode
    task_description = "red cup"  # default
    if mode == "ai":
        try:
            body = await request.json()
            if body and "task" in body:
                task_description = body["task"].strip()
                if not task_description:
                    task_description = "red cup"
        except Exception:
            # If no body or invalid JSON, use default
            pass

    try:
        # Find the main script
        current_dir = Path(__file__).parent.parent  # .../phosphobot/
        kin_dir = current_dir / "ai_kinematics"
        main_script = kin_dir / "main.py"
        if not main_script.exists():
            raise HTTPException(status_code=500, detail="ai_kinematics main.py not found")

        # Launch the subprocess using system Python where record3d is installed
        cmd = [
            "/opt/homebrew/opt/python@3.10/bin/python3.10",
            str(main_script),
            mode,
        ]
        if mode == "ai":
            cmd.append(task_description)  # use user-provided or default task

        # Spawn the process with proper error capture
        env = os.environ.copy()
        # Ensure GUI applications can access the display
        if "DISPLAY" not in env:
            env["DISPLAY"] = ":0"
        
        # Set Anthropic API key for AI features
        if mode == "ai" and _anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = _anthropic_api_key
            logger.info("API key passed to kinematics subprocess")
            logger.info(f"API key length: {len(_anthropic_api_key)}")
            logger.info(f"API key prefix: {_anthropic_api_key[:10]}...")
        elif mode == "ai":
            # This should not happen due to the check above, but just in case
            raise HTTPException(status_code=400, detail="API key not available for AI mode")
        
        # Debug: print environment variables being passed
        if mode == "ai":
            api_key_in_env = env.get("ANTHROPIC_API_KEY")
            if api_key_in_env:
                logger.info(f"Subprocess will receive API key: {api_key_in_env[:10]}...")
            else:
                logger.error("No API key found in subprocess environment!")
        
        kinematics_process = subprocess.Popen(
            cmd,
            cwd=str(kin_dir),
            env=env,  # Pass the environment with DISPLAY and API key
        )
        
        # Give the process a moment to start
        import time
        time.sleep(1.0)
        
        # Check if process is still running
        if kinematics_process.poll() is not None:
            # Process already terminated, capture error
            try:
                stdout, stderr = kinematics_process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "timeout", "timeout"
            
            exit_code = kinematics_process.returncode
            error_msg = f"Process exited with code {exit_code}."
            
            if stderr:
                error_msg += f" STDERR: {stderr.strip()}"
            if stdout:
                error_msg += f" STDOUT: {stdout.strip()}"
                
            # Check for common error patterns
            if "Device connection error" in stdout or "Device connection error" in stderr:
                error_msg += " | Likely cause: No Record3D device connected. Make sure Record3D app is running on iPhone and connected to same network."
            elif "Import error" in stdout or "ModuleNotFoundError" in stderr:
                error_msg += " | Likely cause: Missing Python dependencies. Run 'pip install -r requirements.txt' in ai_kinematics directory."
            
            logger.error(error_msg)
            kinematics_process = None
            raise HTTPException(status_code=500, detail=error_msg)

        logger.info(f"Launched AI-kinematics ({mode}) with PID {kinematics_process.pid}")
        return StatusResponse(status="ok", message=f"AI-kinematics launched in {mode} mode")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to launch AI-kinematics: {e}")
        if kinematics_process:
            try:
                kinematics_process.terminate()
            except:
                pass
            kinematics_process = None
        raise HTTPException(status_code=500, detail=f"Failed to launch kinematics: {e}")


@router.post("/kinematics/stop", response_model=StatusResponse)
async def stop_kinematics() -> StatusResponse:
    """Stop the running AI-kinematics process (if any)."""

    global kinematics_process

    if kinematics_process is None or kinematics_process.poll() is not None:
        return StatusResponse(status="ok", message="No kinematics process is currently running")

    try:
        kinematics_process.terminate()
        try:
            kinematics_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            kinematics_process.kill()
            kinematics_process.wait()

        pid = kinematics_process.pid
        kinematics_process = None
        logger.info(f"Stopped AI-kinematics (PID {pid})")
        return StatusResponse(status="ok", message="Kinematics process stopped successfully")

    except Exception as e:
        logger.error(f"Failed to stop AI-kinematics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop kinematics: {e}")


@router.get("/kinematics/status", response_model=StatusResponse)
async def get_kinematics_status() -> StatusResponse:
    """Return the status of the AI-kinematics subprocess."""

    global kinematics_process

    if kinematics_process is None:
        return StatusResponse(status="ok", message="No kinematics process is currently running")

    if kinematics_process.poll() is None:
        return StatusResponse(status="ok", message=f"Kinematics process is running (PID {kinematics_process.pid})")

    # Process completed – reset handle and try to get exit info
    exit_code = kinematics_process.returncode
    kinematics_process = None
    return StatusResponse(status="ok", message=f"Kinematics process has stopped (exit code: {exit_code})") 