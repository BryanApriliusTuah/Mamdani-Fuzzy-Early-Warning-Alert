from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from main2 import DynamicFuzzyFloodWarningSystem

app = FastAPI(title="Flood Warning API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fuzzy_system = DynamicFuzzyFloodWarningSystem(reading_interval_seconds=1)


class CalibrationRequest(BaseModel):
    ground_distance: float = Field(..., gt=0)
    siaga_level: Optional[float] = Field(None, gt=0)
    banjir_level: Optional[float] = Field(None, gt=0)


class RiskRequest(BaseModel):
    current_distance: float = Field(..., gt=0)
    current_rainfall_mm_per_hour: float = Field(0, ge=0, le=25)


@app.post("/api/calibrate")
async def calibrate(request: CalibrationRequest):
    try:
        if request.siaga_level and request.banjir_level:
            if request.siaga_level <= request.banjir_level:
                raise HTTPException(400, "siaga_level must be > banjir_level")
        
        fuzzy_system.calibrate(
            ground_distance=request.ground_distance,
            siaga_level_override=request.siaga_level,
            banjir_level_override=request.banjir_level
        )
        
        return {
            "success": True,
            "calibration_height": fuzzy_system.calibration_height,
            "siaga_level": fuzzy_system.siaga_level,
            "banjir_level": fuzzy_system.banjir_level
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/calculate-risk")
async def calculate_risk(request: RiskRequest):
    if not fuzzy_system.calibration_height:
        raise HTTPException(400, "System not calibrated. Call /api/calibrate first")
    
    try:
        result = fuzzy_system.calculate_risk(
            current_distance=request.current_distance,
            current_rainfall_mm_per_hour=request.current_rainfall_mm_per_hour
        )
        
        # Calculate additional fields
        water_depth = request.current_distance - fuzzy_system.calibration_height
        
        # Determine risk category
        risk_score = result['risk_score']
        if risk_score < 30:
            risk_category = "Rendah"
        elif risk_score < 65:
            risk_category = "Sedang"
        else:
            risk_category = "Tinggi"
        
        # Time to flood estimation
        avg_rate = result['avg_rate_change_cm_per_min']
        time_to_flood = None
        if avg_rate < -0.1:  # Water rising
            distance_to_flood = request.current_distance - fuzzy_system.banjir_level
            if distance_to_flood > 0:
                time_to_flood = round(distance_to_flood / abs(avg_rate), 1)
        
        # Recovery detection
        is_recovery = (
            result['previous_warning_level'] in ['SIAGA', 'BANJIR'] and
            result['warning_level'] == 'NORMAL'
        )
        
        return {
            "reading_number": result['reading_number'],
            "current_distance": result['current_distance'],
            "water_depth_from_ground": round(water_depth, 2),
            "avg_rate_change_cm_per_min": round(avg_rate, 4),
            "readings_count": len(fuzzy_system.distance_history),
            "water_level_normalized": round(result['water_level_normalized'], 3),
            "avg_rate_normalized": round(result['avg_rate_normalized'], 3),
            "rainfall_normalized": round(result['rainfall_normalized'], 3),
            "risk_score": round(risk_score, 1),
            "risk_category": risk_category,
            "warning_level": result['warning_level'],
            "previous_warning_level": result['previous_warning_level'],
			"status_mesage": result['status_message'],
            "is_recovery": is_recovery,
            "time_to_flood_minutes": time_to_flood
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/status")
async def get_status():
    return {
        "is_calibrated": fuzzy_system.calibration_height is not None,
        "calibration_height": fuzzy_system.calibration_height,
        "siaga_level": fuzzy_system.siaga_level,
        "banjir_level": fuzzy_system.banjir_level,
        "reading_interval_seconds": fuzzy_system.reading_interval_seconds,
        "readings_count": len(fuzzy_system.distance_history),
        "previous_warning_level": fuzzy_system.previous_warning_level
    }


@app.post("/api/reset")
async def reset():
    if not fuzzy_system.calibration_height:
        raise HTTPException(400, "System not calibrated")
    
    fuzzy_system.reset_history()
    return {"success": True, "message": "History reset successfully"}


@app.get("/")
async def root():
    return {
        "message": "Flood Warning API v2.0",
        "endpoints": {
            "calibrate": "/api/calibrate",
            "calculate_risk": "/api/calculate-risk",
            "status": "/api/status",
            "reset": "/api/reset"
        }
    }


if __name__ == "__main__":
    print("🌊 Flood Warning API v2.0")
    print("📡 http://localhost:8000")
    print("📚 http://localhost:8000/docs")
    uvicorn.run("api2:app", host="0.0.0.0", port=8000, reload=True)