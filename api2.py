from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
import uvicorn

from main2 import DynamicFuzzyFloodWarningSystem

app = FastAPI(
    title="Flood Early Warning Fuzzy Logic API",
    description="API for flood monitoring with 60-second average rate of change and recovery detection",
    version="3.0.0"
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
    current_distance: float = Field(..., description="Distance from sensor to water surface (cm)", example=165.0, gt=0)
    current_rainfall_mm_per_hour: float = Field(default=0, description="Rainfall intensity (mm/hour)", example=12.5, ge=0, le=25)
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO format timestamp (optional, defaults to current time)",
        example="2025-10-07T10:30:00"
    )

class RiskCalculationResponse(BaseModel):
    current_distance: float
    water_depth_from_ground: float
    avg_rate_change_cm_per_min: float = Field(..., description="Average rate of change over last 60 seconds (cm/min)")
    readings_count: int = Field(..., description="Number of readings used for averaging")
    water_level_normalized: float
    avg_rate_normalized: float
    avg_rate_category: str = Field(..., description="Categorized rate of change (e.g., Naik Sangat Cepat, Stabil, Turun Cepat)")
    current_rainfall: float
    rainfall_normalized: float
    current_rainfall_category: str
    risk_score: float
    risk_category: str = Field(..., description="Categorized risk level (Rendah, Sedang, Tinggi)")
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

class BatchReadingItem(BaseModel):
    current_distance: float = Field(..., description="Distance from sensor to water surface (cm)", example=165.0, gt=0)
    current_rainfall_mm_per_hour: float = Field(default=0, description="Rainfall intensity (mm/hour)", ge=0, le=25)
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO format timestamp",
        example="2025-10-07T10:30:00"
    )

class BatchReadingResult(BaseModel):
    index: int
    current_distance: float
    water_depth_from_ground: float
    avg_rate_change_cm_per_min: float
    avg_rate_category: str = Field(..., description="Categorized rate of change")
    risk_score: float
    risk_category: str = Field(..., description="Categorized risk level")
    warning_level: str
    time_to_flood_minutes: Optional[float]
    status_message: str

class BatchReadingsRequest(BaseModel):
    readings: List[BatchReadingItem] = Field(..., description="List of sensor readings")

class BatchReadingsResponse(BaseModel):
    success: bool
    timestamp: str
    total_readings: int
    results: List[BatchReadingResult]

class SystemStatusResponse(BaseModel):
    is_calibrated: bool
    calibration_height: Optional[float]
    siaga_level: Optional[float]
    banjir_level: Optional[float]
    previous_warning_level: Optional[str] = Field(None, description="Last recorded warning level")
    readings_count: int = Field(..., description="Number of readings in history")
    history_size: int = Field(..., description="Maximum history capacity (60 seconds)")

class ThresholdsResponse(BaseModel):
    calibration_height: float
    banjir_level: float
    siaga_level: float
    distance_range: Dict[str, str]
    notification_intervals: Dict[str, Optional[str]]

class SimulationRequest(BaseModel):
    scenario: str = Field(..., description="Scenario type: rising, falling, or stable", example="rising")
    duration_seconds: int = Field(default=120, description="Duration of simulation in seconds", example=120)
    interval_seconds: int = Field(default=6, description="Interval between readings in seconds", example=6)
    start_distance: float = Field(default=200.0, description="Starting distance in cm", example=200.0)
    rainfall: float = Field(default=12.5, description="Rainfall intensity (mm/hour)", example=12.5)

class SimulationResult(BaseModel):
    time_seconds: int
    timestamp: str
    current_distance: float
    water_depth: float
    avg_rate_change: float
    avg_rate_category: str
    risk_score: float
    risk_category: str
    warning_level: str
    time_to_flood: Optional[float]
    message: str

class SimulationResponse(BaseModel):
    success: bool
    scenario: str
    parameters: Dict
    total_readings: int
    results: List[SimulationResult]

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Flood Early Warning Fuzzy Logic API v3.0",
        "description": "3-input fuzzy: Water Level + Average Rate of Change (60s, cm/min) + Rainfall",
        "features": [
            "60-second moving average for rate of change calculation",
            "Time-to-flood estimation", 
            "Real-time risk assessment",
            "Recovery detection (SIAGA/BANJIR → NORMAL)",
            "Automatic recovery notifications",
            "Guideline-based rate calibration (0.067 cm/min)"
        ],
        "endpoints": {
            "calibrate": "/api/calibrate",
            "calculate_risk": "/api/calculate-risk",
            "batch_readings": "/api/batch-readings",
            "status": "/api/status",
            "thresholds": "/api/thresholds",
            "reset": "/api/reset",
            "simulate": "/api/simulate",
            "docs": "/docs"
        }
    }

@app.get("/api/status", response_model=SystemStatusResponse, tags=["System"])
async def get_system_status():
    """
    Get current system status including calibration, warning levels, and history
    """
    return {
        "is_calibrated": fuzzy_system.calibration_height is not None,
        "calibration_height": fuzzy_system.calibration_height,
        "siaga_level": fuzzy_system.siaga_level,
        "banjir_level": fuzzy_system.banjir_level,
        "previous_warning_level": fuzzy_system.previous_warning_level,
        "readings_count": len(fuzzy_system.distance_history),
        "history_size": 60
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
    Calculate flood risk using 60-second average rate of change
    
    **Inputs:**
    1. Water Level (current_distance): cm from sensor to water surface
    2. Average Rate of Change: Automatically calculated from last 60 seconds of readings (cm/min)
       - Negative = water rising (distance decreasing)
       - Positive = water falling (distance increasing)
    3. Current Rainfall (current_rainfall_mm_per_hour): mm/hour (0-25)
    
    **How it works:**
    The system maintains a 60-second rolling history of distance readings. When you submit
    a new reading, it automatically calculates the average rate of change over the available
    history (up to 60 seconds). This provides a more stable and accurate trend compared to
    instantaneous measurements.
    
    **Outputs:**
    - Risk assessment with warning level (NORMAL, SIAGA, BANJIR)
    - Average rate of change over last 60 seconds (cm/min)
    - Categorized rate of change (e.g., Naik Sangat Cepat, Stabil, Turun Cepat)
    - Number of readings used for averaging
    - Risk score and categorized risk level (Rendah, Sedang, Tinggi)
    - Estimated time until flood (for evacuation planning)
    - Recovery detection (when status returns to NORMAL from SIAGA/BANJIR)
    - Detailed status message in Indonesian
    
    **Recovery Detection:**
    The system automatically detects when conditions improve from SIAGA or BANJIR 
    back to NORMAL, setting `is_recovery=True` and `should_send_recovery_notification=True`
    to enable sending "all clear" notifications to users.
    
    **Rate Guideline:**
    The membership functions are calibrated based on a guideline average rate of 
    0.0673611 cm/min (approximately 4 cm/hour).
    """
    if fuzzy_system.calibration_height is None:
        raise HTTPException(status_code=400, detail="System not calibrated. Call /api/calibrate first")
    
    try:
        # Parse timestamp if provided
        timestamp = None
        if request.timestamp:
            try:
                timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
        
        result = fuzzy_system.calculate_risk(
            current_distance=request.current_distance,
            current_rainfall_mm_per_hour=request.current_rainfall_mm_per_hour,
            timestamp=timestamp
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")

@app.post("/api/batch-readings", response_model=BatchReadingsResponse, tags=["Risk Assessment"])
async def add_batch_readings(request: BatchReadingsRequest):
    """
    Add multiple sensor readings at once (for historical data or bulk processing)
    
    This endpoint is useful for:
    - Loading historical data into the system
    - Batch processing of accumulated readings
    - Simulation and testing scenarios
    
    The system will process each reading in order and calculate the average rate
    of change based on the cumulative history.
    """
    if fuzzy_system.calibration_height is None:
        raise HTTPException(status_code=400, detail="System not calibrated. Call /api/calibrate first")
    
    if not request.readings or len(request.readings) == 0:
        raise HTTPException(status_code=400, detail="readings array cannot be empty")
    
    try:
        results = []
        
        for idx, reading in enumerate(request.readings):
            # Parse timestamp if provided
            timestamp = None
            if reading.timestamp:
                try:
                    timestamp = datetime.fromisoformat(reading.timestamp.replace('Z', '+00:00'))
                except ValueError:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid timestamp format in reading at index {idx}"
                    )
            
            # Calculate flood risk
            result = fuzzy_system.calculate_risk(
                current_distance=reading.current_distance,
                current_rainfall_mm_per_hour=reading.current_rainfall_mm_per_hour,
                timestamp=timestamp
            )
            
            results.append(BatchReadingResult(
                index=idx,
                current_distance=result['current_distance'],
                water_depth_from_ground=round(result['water_depth_from_ground'], 2),
                avg_rate_change_cm_per_min=round(result['avg_rate_change_cm_per_min'], 4),
                avg_rate_category=result['avg_rate_category'],
                risk_score=round(result['risk_score'], 1),
                risk_category=result['risk_category'],
                warning_level=result['warning_level'],
                time_to_flood_minutes=result['time_to_flood_minutes'],
                status_message=result['status_message']
            ))
        
        return BatchReadingsResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            total_readings=len(results),
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/api/thresholds", response_model=ThresholdsResponse, tags=["System"])
async def get_thresholds():
    """
    Get current warning level thresholds and distance ranges
    """
    if fuzzy_system.calibration_height is None:
        raise HTTPException(status_code=400, detail="System not calibrated")
    
    return {
        "calibration_height": fuzzy_system.calibration_height,
        "banjir_level": fuzzy_system.banjir_level,
        "siaga_level": fuzzy_system.siaga_level,
        "distance_range": {
            "safe_zone": f"> {fuzzy_system.siaga_level} cm",
            "siaga_zone": f"{fuzzy_system.banjir_level} - {fuzzy_system.siaga_level} cm",
            "banjir_zone": f"<= {fuzzy_system.banjir_level} cm"
        },
        "notification_intervals": {
            "NORMAL": None,
            "SIAGA": "10 minutes",
            "BANJIR": "5 minutes"
        }
    }

@app.post("/api/reset", tags=["System"])
async def reset_system():
    """
    Reset the system history (clears all readings and warning state)
    
    This maintains the calibration but clears:
    - All distance reading history
    - All timestamp history
    - Previous warning level
    
    Use this when you want to start fresh monitoring without recalibrating.
    """
    if fuzzy_system.calibration_height is None:
        raise HTTPException(status_code=400, detail="System not calibrated")
    
    try:
        # Clear history
        fuzzy_system.distance_history.clear()
        fuzzy_system.timestamp_history.clear()
        fuzzy_system.previous_warning_level = None
        
        return {
            "success": True, 
            "message": "System history reset successfully. Calibration maintained.",
            "calibration_maintained": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.post("/api/simulate", response_model=SimulationResponse, tags=["Simulation"])
async def simulate_scenario(request: SimulationRequest):
    """
    Simulate a flood scenario with generated data
    
    **Scenarios:**
    - `rising`: Water level rising (distance decreasing by 0.3 cm per interval)
    - `falling`: Water level falling (distance increasing by 0.3 cm per interval)
    - `stable`: Water level stable with small random variations (±0.1 cm)
    
    **Parameters:**
    - `duration_seconds`: Total duration of the simulation
    - `interval_seconds`: Time between readings (e.g., 6 seconds)
    - `start_distance`: Initial distance from sensor to water
    - `rainfall`: Constant rainfall intensity throughout simulation
    
    This is useful for:
    - Testing the system behavior under different scenarios
    - Training and demonstration purposes
    - Understanding how the 60-second averaging works
    """
    if fuzzy_system.calibration_height is None:
        raise HTTPException(status_code=400, detail="System not calibrated. Call /api/calibrate first")
    
    if request.scenario not in ['rising', 'falling', 'stable']:
        raise HTTPException(
            status_code=400, 
            detail="Invalid scenario. Must be: 'rising', 'falling', or 'stable'"
        )
    
    try:
        import numpy as np
        from datetime import timedelta
        
        # Generate simulated readings
        num_readings = request.duration_seconds // request.interval_seconds
        base_time = datetime.now()
        
        results = []
        
        for i in range(num_readings):
            timestamp = base_time + timedelta(seconds=i * request.interval_seconds)
            
            # Calculate distance based on scenario
            if request.scenario == 'rising':
                # Water rising (distance decreasing)
                current_distance = request.start_distance - (i * 0.3)
            elif request.scenario == 'falling':
                # Water falling (distance increasing)
                current_distance = request.start_distance + (i * 0.3)
            else:  # stable
                # Water stable with small variations
                current_distance = request.start_distance + np.random.uniform(-0.1, 0.1)
            
            # Calculate flood risk
            result = fuzzy_system.calculate_risk(
                current_distance=current_distance,
                current_rainfall_mm_per_hour=request.rainfall,
                timestamp=timestamp
            )
            
            results.append(SimulationResult(
                time_seconds=i * request.interval_seconds,
                timestamp=timestamp.isoformat(),
                current_distance=round(result['current_distance'], 2),
                water_depth=round(result['water_depth_from_ground'], 2),
                avg_rate_change=round(result['avg_rate_change_cm_per_min'], 4),
                avg_rate_category=result['avg_rate_category'],
                risk_score=round(result['risk_score'], 1),
                risk_category=result['risk_category'],
                warning_level=result['warning_level'],
                time_to_flood=result['time_to_flood_minutes'],
                message=result['status_message']
            ))
        
        return SimulationResponse(
            success=True,
            scenario=request.scenario,
            parameters={
                "duration_seconds": request.duration_seconds,
                "interval_seconds": request.interval_seconds,
                "start_distance": request.start_distance,
                "rainfall": request.rainfall
            },
            total_readings=len(results),
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.get("/health", tags=["General"])
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "is_calibrated": fuzzy_system.calibration_height is not None,
        "api_version": "3.0.0",
        "features": {
            "average_calculation": "60-second rolling window",
            "rate_unit": "cm/min",
            "guideline_rate": "0.067 cm/min"
        }
    }

if __name__ == "__main__":
    print("=" * 70)
    print("Flood Early Warning Fuzzy Logic API v3.0")
    print("=" * 70)
    print("\n✨ Features:")
    print("  • 60-second moving average for rate of change")
    print("  • Time-to-flood estimation")
    print("  • Recovery detection (SIAGA/BANJIR → NORMAL)")
    print("  • Guideline-based calibration (0.067 cm/min)")
    print("\n📡 Endpoints:")
    print("  • POST /api/calibrate       - Calibrate system")
    print("  • POST /api/calculate-risk  - Calculate flood risk")
    print("  • POST /api/batch-readings  - Process multiple readings")
    print("  • GET  /api/status          - Get system status")
    print("  • GET  /api/thresholds      - Get warning thresholds")
    print("  • POST /api/reset           - Reset system history")
    print("  • POST /api/simulate        - Run simulation")
    print("  • GET  /health              - Health check")
    print("\n🌐 Starting server...")
    print("  • API: http://localhost:8000")
    print("  • Docs: http://localhost:8000/docs")
    print("  • ReDoc: http://localhost:8000/redoc")
    print("=" * 70)
    uvicorn.run("api2:app", host="0.0.0.0", port=8000, reload=True)