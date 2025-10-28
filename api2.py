from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from main2 import DynamicFuzzyFloodWarningSystem

app = FastAPI(title="Flood Warning API", version="2.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fuzzy_system = DynamicFuzzyFloodWarningSystem(reading_interval_seconds=1)


class CalibrationRequest(BaseModel):
    ground_distance: float = Field(..., gt=0, description="Distance from sensor to ground (cm)")
    siaga_level: Optional[float] = Field(None, gt=0, description="Distance threshold for Siaga I level (cm)")
    banjir_level: Optional[float] = Field(None, gt=0, description="Distance threshold for flood level (cm)")


class RiskRequest(BaseModel):
    current_distance: float = Field(..., ge=0, description="Current distance reading from sensor (cm)")
    current_rainfall_mm_per_hour: float = Field(0, ge=0, le=21, description="Current rainfall intensity (mm/hour)")


@app.post("/api/calibrate")
async def calibrate(request: CalibrationRequest):
    try:
        # Validate relationship between levels
        if request.siaga_level and request.banjir_level:
            if request.siaga_level <= request.banjir_level:
                raise HTTPException(
                    status_code=400, 
                    detail="siaga_level must be greater than banjir_level"
                )
        
        fuzzy_system.calibrate(
            ground_distance=request.ground_distance,
            siaga_level_override=request.siaga_level,
            banjir_level_override=request.banjir_level
        )
        
        return {
            "success": True,
            "message": "System calibrated successfully",
            "calibration_height": fuzzy_system.calibration_height,
            "siaga_level": fuzzy_system.siaga_level,
            "banjir_level": fuzzy_system.banjir_level,
            "threshold_range_cm": fuzzy_system.siaga_level - fuzzy_system.banjir_level
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@app.post("/api/calculate-risk")
async def calculate_risk(request: RiskRequest):
    if not fuzzy_system.calibration_height:
        raise HTTPException(
            status_code=400, 
            detail="System not calibrated. Please call /api/calibrate first"
        )
    
    try:
        result = fuzzy_system.calculate_risk(
            current_distance=request.current_distance,
            current_rainfall_mm_per_hour=request.current_rainfall_mm_per_hour
        )
        
        water_depth = request.current_distance - fuzzy_system.calibration_height
    
        distance_to_banjir = request.current_distance - fuzzy_system.banjir_level
        distance_to_siaga2 = request.current_distance - (
            (fuzzy_system.siaga_level + fuzzy_system.banjir_level) / 2
        )
        distance_to_siaga1 = request.current_distance - fuzzy_system.siaga_level
        
        avg_rate = result['avg_rate_change_cm_per_min']
        time_to_flood = None
        time_to_siaga2 = None
        time_to_siaga1 = None
        
        if avg_rate > 0.1: 
            if distance_to_banjir > 0:
                time_to_flood = round(distance_to_banjir / abs(avg_rate), 1)
            if distance_to_siaga2 > 0:
                time_to_siaga2 = round(distance_to_siaga2 / abs(avg_rate), 1)
            if distance_to_siaga1 > 0:
                time_to_siaga1 = round(distance_to_siaga1 / abs(avg_rate), 1)
        
        is_recovery = (
            result['previous_warning_level'] in ['MODERATE', 'HIGH', 'VERY HIGH'] and
            result['warning_level'] in ['VERY LOW', 'LOW']
        )
        
        is_rapid_rise = avg_rate > 5 * fuzzy_system.RATE_FACTOR
        is_rapid_fall = avg_rate < -5 * fuzzy_system.RATE_FACTOR
        
        return {
            "success": True,
            "reading_number": result['reading_number'],
            "timestamp_info": {
                "readings_count": len(fuzzy_system.distance_history),
                "reading_interval_seconds": fuzzy_system.reading_interval_seconds,
            },
            "sensor_data": {
                "current_distance_cm": round(result['current_distance'], 2),
                "water_depth_from_ground_cm": round(water_depth, 2),
                "current_rainfall_mm_per_hour": request.current_rainfall_mm_per_hour,
            },
            "water_level_status": {
                "distance_to_banjir_cm": round(distance_to_banjir, 2),
                "distance_to_siaga2_cm": round(distance_to_siaga2, 2),
                "distance_to_siaga1_cm": round(distance_to_siaga1, 2),
            },
            "rate_of_change": {
                "avg_rate_cm_per_sec": round(result['rate_change_cm_per_sec'], 4),
                "avg_rate_cm_per_min": round(avg_rate, 4),
                "is_rapid_rise": is_rapid_rise,
                "is_rapid_fall": is_rapid_fall,
            },
            "risk_assessment": {
                "risk_score": round(result['risk_score'], 2),
                "risk_category": result['risk_category'],
                "warning_level": result['warning_level'],
                "previous_warning_level": result['previous_warning_level'],
                "status_message": result['status_message'],
                "is_recovery": is_recovery,
            },
            "predictions": {
                "time_to_flood_minutes": time_to_flood,
                "time_to_siaga2_minutes": time_to_siaga2,
                "time_to_siaga1_minutes": time_to_siaga1,
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")


@app.get("/api/status")
async def get_status():
    """
    Get current system status and calibration info.
    """
    return {
        "is_calibrated": fuzzy_system.calibration_height is not None,
        "calibration_height": fuzzy_system.calibration_height,
        "siaga_level": fuzzy_system.siaga_level,
        "banjir_level": fuzzy_system.banjir_level,
        "threshold_range_cm": (
            fuzzy_system.siaga_level - fuzzy_system.banjir_level 
            if fuzzy_system.siaga_level and fuzzy_system.banjir_level 
            else None
        ),
        "reading_interval_seconds": fuzzy_system.reading_interval_seconds,
        "readings_count": len(fuzzy_system.distance_history),
        "previous_warning_level": fuzzy_system.previous_warning_level,
        "rate_factor": fuzzy_system.RATE_FACTOR
    }


@app.get("/api/system-info")
async def get_system_info():
    """
    Get detailed system information using the get_system_info() method.
    """
    try:
        return fuzzy_system.get_system_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system info: {str(e)}")


@app.post("/api/reset")
async def reset():
    """
    Reset reading history while maintaining calibration.
    """
    if not fuzzy_system.calibration_height:
        raise HTTPException(
            status_code=400, 
            detail="System not calibrated. Nothing to reset."
        )
    
    fuzzy_system.reset_history()
    return {
        "success": True, 
        "message": "Reading history reset successfully. Calibration maintained."
    }


@app.get("/api/thresholds")
async def get_thresholds():
    """
    Get all warning level thresholds.
    """
    if not fuzzy_system.calibration_height:
        raise HTTPException(
            status_code=400,
            detail="System not calibrated"
        )
    
    mid_point = (fuzzy_system.siaga_level + fuzzy_system.banjir_level) / 2
    
    return {
        "calibration_height": fuzzy_system.calibration_height,
        "thresholds": {
            "normal": {
                "distance_cm": f"> {fuzzy_system.siaga_level}",
                "description": "Water level is normal"
            },
            "siaga_1": {
                "distance_range_cm": f"{mid_point} - {fuzzy_system.siaga_level}",
                "description": "Alert level 1 - Water level elevated"
            },
            "siaga_2": {
                "distance_range_cm": f"{fuzzy_system.banjir_level} - {mid_point}",
                "description": "Alert level 2 - Water approaching flood level"
            },
            "banjir": {
                "distance_cm": f"≤ {fuzzy_system.banjir_level}",
                "description": "Flood level - Critical water level"
            }
        }
    }


@app.get("/")
async def root():
    """
    API root endpoint with documentation links.
    """
    return {
        "message": "Flood Warning API v2.0.1",
        "description": "Dynamic Fuzzy Logic Flood Warning System",
        "documentation": "/docs",
        "endpoints": {
            "calibrate": {
                "method": "POST",
                "path": "/api/calibrate",
                "description": "Calibrate the system with ground distance and thresholds"
            },
            "calculate_risk": {
                "method": "POST",
                "path": "/api/calculate-risk",
                "description": "Calculate flood risk from sensor reading"
            },
            "status": {
                "method": "GET",
                "path": "/api/status",
                "description": "Get current system status"
            },
            "system_info": {
                "method": "GET",
                "path": "/api/system-info",
                "description": "Get detailed system information"
            },
            "thresholds": {
                "method": "GET",
                "path": "/api/thresholds",
                "description": "Get warning level thresholds"
            },
            "reset": {
                "method": "POST",
                "path": "/api/reset",
                "description": "Reset reading history"
            }
        }
    }


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "version": "2.0.1",
        "system_calibrated": fuzzy_system.calibration_height is not None
    }


if __name__ == "__main__":
    print("🌊 Flood Warning API v2.0.1")
    print("📡 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🔧 ReDoc: http://localhost:8000/redoc")
    uvicorn.run("api2:app", host="0.0.0.0", port=8000, reload=True)