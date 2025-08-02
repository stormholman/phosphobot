from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from loguru import logger

from phosphobot.models import StatusResponse

router = APIRouter(tags=["kinematics"])

# Global variable to track the AI-kinematics subprocess
kinematics_process: Optional[subprocess.Popen[str]] = None


@router.post("/kinematics/launch", response_model=StatusResponse)
async def launch_kinematics(
    background_tasks: BackgroundTasks,
    mode: Literal["manual", "ai"] = Query("manual", description="Operating mode for the kinematics app"),
) -> StatusResponse:
    """Launch the AI-kinematics stream processing in *manual* or *AI* mode.

    The underlying script is *phosphobot/ai_kinematics/main.py* which now accepts
    command line arguments for non-interactive operation.
    """

    global kinematics_process

    # Prevent double-launch
    if kinematics_process is not None and kinematics_process.poll() is None:
        raise HTTPException(status_code=400, detail="Kinematics process already running")

    # Clean up any finished process handle
    kinematics_process = None

    try:
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
            cmd.append("red cup")  # default task for AI mode

        # Spawn the process with proper error capture
        kinematics_process = subprocess.Popen(
            cmd,
            cwd=str(kin_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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