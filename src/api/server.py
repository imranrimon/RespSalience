import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import uvicorn
import shutil
from src.core.diagnosis import ARIDiagnostic

app = FastAPI(title="Respiratory Monitoring Telemedicine API")

@app.post("/diagnose")
async def diagnose_respiration(
    age_months: int = Form(...),
    video: UploadFile = File(...)
):
    """
    Receives a video file and infant age, returns respiration rate and ARI diagnosis.
    """
    # 1. Save video temporarily
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, video.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    # 2. Mocking the inference result for now
    mock_bpm = 55.0 
    
    # 3. Perform ARI diagnosis
    diagnostic_result = ARIDiagnostic.classify_rr(mock_bpm, age_months)
    
    # 4. Clean up
    os.remove(temp_path)
    
    return {
        "filename": video.filename,
        "diagnosis": diagnostic_result,
        "recommendation": "Consult a pediatrician immediately." if diagnostic_result["is_tachypnea"] else "Continue monitoring."
    }

@app.post("/vitals-check")
async def check_vitals(
    age_months: int = Form(...),
    bpm: float = Form(...),
    oxygen_saturation: Optional[float] = Form(None)
):
    """
    Receives numeric vitals and returns ARI diagnosis.
    Useful for simulation and external sensor integration.
    """
    diagnostic_result = ARIDiagnostic.classify_rr(bpm, age_months)
    
    response = {
        "diagnosis": diagnostic_result,
        "oxygen_saturation": oxygen_saturation,
        "recommendation": "Consult a pediatrician immediately." if diagnostic_result["is_tachypnea"] else "Continue monitoring."
    }
    
    if oxygen_saturation is not None and oxygen_saturation < 95:
        response["warning"] = "Low oxygen saturation detected!"
        
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
