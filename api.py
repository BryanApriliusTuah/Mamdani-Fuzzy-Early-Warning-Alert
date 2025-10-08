from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
import uvicorn

from main import DynamicFuzzyFloodWarningSystem

app = FastAPI(
    title="Flood Early Warning Fuzzy Logic API",
    description="API for flood monitoring with time-to-flood estimation and recovery detection",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fuzzy_system = DynamicFuzzyFloodWarningSystem()

class CalibrationRequest(BaseModel):
    ground_distance: float = Field(..., description="Distance from sensor to ground (cm)", example=156.59, gt=0)

class CalibrationResponse(BaseModel):
    success: bool
    message: str
    calibration_height: float
    siaga_level: float
    banjir_level: float

class RiskCalculationRequest(BaseModel):
    current_distance: float = Field(..., description="Distance from sensor to water surface (cm)", example=160.5, gt=0)
    current_rainfall_mm_per_hour: float = Field(default=0, description="Rainfall intensity (mm/hour)", example=12.5, ge=0, le=25)
    water_elevation_change_cm_per_sec: Optional[float] = Field(
        default=None, 
        description="Rate of water level change (cm/sec). Negative=rising, Positive=falling", 
        example=-0.05
    )

class RiskCalculationResponse(BaseModel):
    current_distance: float
    water_depth_from_ground: float
    rate_of_change_cm_per_sec: float
    rate_of_change_cm_per_min: float
    water_level_normalized: float
    rate_change_normalized: float
    current_rainfall: float
    rainfall_normalized: float
    current_rainfall_category: str
    risk_score: float
    warning_level: str
    previous_warning_level: Optional[str] = Field(None, description="Previous warning level (for tracking changes)")
    is_recovery: bool = Field(..., description="True if status recovered from SIAGA/BANJIR to NORMAL")
    notification_interval: Optional[int]
    should_send_warning: bool
    should_send_recovery_notification: bool = Field(..., description="True if recovery notification should be sent")
    time_to_flood_minutes: Optional[float]
    time_to_flood_status: str
    status_message: str
    thresholds: Dict[str, float]

class SystemStatusResponse(BaseModel):
    is_calibrated: bool
    calibration_height: Optional[float]
    siaga_level: Optional[float]
    banjir_level: Optional[float]
    previous_warning_level: Optional[str] = Field(None, description="Last recorded warning level")

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Flood Early Warning Fuzzy Logic API v2.1",
        "description": "3-input fuzzy: Water Level + Rate of Change (cm/sec) + Rainfall",
        "features": [
            "Time-to-flood estimation", 
            "Real-time risk assessment",
            "Recovery detection (SIAGA/BANJIR → NORMAL)",
            "Automatic recovery notifications"
        ],
        "endpoints": {
            "calibrate": "/api/calibrate",
            "calculate_risk": "/api/calculate-risk",
            "status": "/api/status",
            "reset": "/api/reset",
            "docs": "/docs"
        }
    }

@app.get("/api/status", response_model=SystemStatusResponse, tags=["System"])
async def get_system_status():
    """
    Get current system status including calibration and warning levels
    """
    return {
        "is_calibrated": fuzzy_system.calibration_height is not None,
        "calibration_height": fuzzy_system.calibration_height,
        "siaga_level": fuzzy_system.siaga_level,
        "banjir_level": fuzzy_system.banjir_level,
        "previous_warning_level": fuzzy_system.previous_warning_level
    }

@app.post("/api/calibrate", response_model=CalibrationResponse, tags=["System"])
async def calibrate_system(request: CalibrationRequest):
    """
    Calibrate the flood warning system with ground distance measurement
    
    This sets the baseline for all future measurements:
    - BANJIR level = ground_distance (flood threshold)
    - SIAGA level = ground_distance + 30cm (warning threshold)
    """
    try:
        fuzzy_system.calibrate(request.ground_distance)
        return {
            "success": True,
            "message": "System calibrated successfully",
            "calibration_height": fuzzy_system.calibration_height,
            "siaga_level": fuzzy_system.siaga_level,
            "banjir_level": fuzzy_system.banjir_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")

@app.post("/api/calculate-risk", response_model=RiskCalculationResponse, tags=["Risk Assessment"])
async def calculate_flood_risk(request: RiskCalculationRequest):
    """
    Calculate flood risk with time-to-flood estimation and recovery detection
    
    **Inputs:**
    1. Water Level (current_distance): cm from sensor to water surface
    2. Rate of Change (water_elevation_change_cm_per_sec): cm/sec (negative=rising, positive=falling)
    3. Current Rainfall (current_rainfall_mm_per_hour): mm/hour (0-25)
    
    **Outputs:**
    - Risk assessment with warning level (NORMAL, SIAGA, BANJIR)
    - Estimated time until flood (for evacuation planning)
    - Recovery detection (when status returns to NORMAL from SIAGA/BANJIR)
    - Detailed status message in Indonesian
    
    **Recovery Detection:**
    The system automatically detects when conditions improve from SIAGA or BANJIR 
    back to NORMAL, setting `is_recovery=True` and `should_send_recovery_notification=True`
    to enable sending "all clear" notifications to users.
    """
    if fuzzy_system.calibration_height is None:
        raise HTTPException(status_code=400, detail="System not calibrated. Call /api/calibrate first")
    
    try:
        result = fuzzy_system.calculate_risk(
            current_distance=request.current_distance,
            current_rainfall_mm_per_hour=request.current_rainfall_mm_per_hour,
            water_elevation_change_cm_per_sec=request.water_elevation_change_cm_per_sec
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")

@app.post("/api/reset", tags=["System"])
async def reset_system():
    """
    Reset the fuzzy logic system to initial state
    
    Clears all calibration data and warning history.
    You must call /api/calibrate again after reset.
    """
    global fuzzy_system
    fuzzy_system = DynamicFuzzyFloodWarningSystem()
    return {
        "success": True, 
        "message": "System reset successfully. Please calibrate before using."
    }

@app.get("/health", tags=["General"])
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "is_calibrated": fuzzy_system.calibration_height is not None,
        "api_version": "2.1.0"
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)