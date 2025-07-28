import asyncio
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from phosphobot.models import StatusResponse

router = APIRouter(tags=["simulation"])

# Global variable to track the simulation process
simulation_process: Optional[subprocess.Popen] = None


@router.post("/simulation/launch", response_model=StatusResponse)
async def launch_simulation(background_tasks: BackgroundTasks):
    """
    Launch the MuJoCo simulation in the background.
    """
    global simulation_process
    
    if simulation_process is not None:
        # Check if the process is still running
        if simulation_process.poll() is None:
            raise HTTPException(status_code=400, detail="Simulation is already running")
        else:
            # Process has ended, clean up
            simulation_process = None
    
    try:
        # Get the path to the mujoco directory
        current_dir = Path(__file__).parent.parent
        mujoco_dir = current_dir / "mujoco"
        main_script = mujoco_dir / "main.py"
        
        if not main_script.exists():
            raise HTTPException(status_code=500, detail="MuJoCo simulation script not found")
        
        # Launch the simulation using mjpython
        simulation_process = subprocess.Popen(
            ["mjpython", str(main_script)],
            cwd=str(mujoco_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Launched MuJoCo simulation with PID: {simulation_process.pid}")
        
        return StatusResponse(
            status="ok",
            message="MuJoCo simulation launched successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to launch simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to launch simulation: {str(e)}")


@router.post("/simulation/stop", response_model=StatusResponse)
async def stop_simulation():
    """
    Stop the running MuJoCo simulation.
    """
    global simulation_process
    
    if simulation_process is None or simulation_process.poll() is not None:
        return StatusResponse(
            status="ok",
            message="No simulation is currently running"
        )
    
    try:
        # Terminate the process gracefully
        simulation_process.terminate()
        
        # Wait a bit for graceful shutdown
        try:
            simulation_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate gracefully
            simulation_process.kill()
            simulation_process.wait()
        
        simulation_process = None
        logger.info("MuJoCo simulation stopped")
        
        return StatusResponse(
            status="ok",
            message="Simulation stopped successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to stop simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop simulation: {str(e)}")


@router.get("/simulation/status", response_model=StatusResponse)
async def get_simulation_status():
    """
    Get the current status of the MuJoCo simulation.
    """
    global simulation_process
    
    if simulation_process is None:
        return StatusResponse(
            status="ok",
            message="No simulation is currently running"
        )
    
    # Check if the process is still running
    if simulation_process.poll() is None:
        return StatusResponse(
            status="ok",
            message=f"Simulation is running (PID: {simulation_process.pid})"
        )
    else:
        # Process has ended
        simulation_process = None
        return StatusResponse(
            status="ok",
            message="Simulation has stopped"
        ) 