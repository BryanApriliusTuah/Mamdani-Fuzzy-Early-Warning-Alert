from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
import uvicorn

from main2 import DynamicFuzzyFloodWarningSystem

app = FastAPI(
    title="Flood Early Warning Fuzzy Logic API",
    description="API for flood monitoring with FIFO queue and dynamic fuzzy categorization",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system with configurable reading interval
fuzzy_system = DynamicFuzzyFloodWarningSystem(reading_interval_seconds=1)

class CalibrationRequest(BaseModel):
    ground_distance: float = Field(..., description="Distance from sensor to ground (cm)", example=156.59, gt=0)
    siaga_level: Optional[float] = Field(None, description="Override siaga level (cm). If not provided, defaults to ground_distance + 30", example=186.59, gt=0)
    banjir_level: Optional[float] = Field(None, description="Override banjir level (cm). If not provided, defaults to ground_distance", example=156.59, gt=0)
    reading_interval_seconds: Optional[int] = Field(1, description="Time interval between readings in seconds", example=1, gt=0)

class CalibrationResponse(BaseModel):
    success: bool
    message: str
    calibration_height: float
    siaga_level: float
    banjir_level: float
    reading_interval_seconds: int
    is_overridden: bool

class RiskCalculationRequest(BaseModel):
    current_distance: float = Field(..., description="Distance from sensor to water surface (cm)", example=165.0, gt=0)
    current_rainfall_mm_per_hour: float = Field(default=0, description="Rainfall intensity (mm/hour)", example=12.5, ge=0, le=25)

class RiskCalculationResponse(BaseModel):
    reading_number: int
    current_distance: float
    water_depth_from_ground: float
    avg_rate_change_cm_per_min: float = Field(..., description="Average rate of change over readings (cm/min)")
    readings_count: int = Field(..., description="Number of readings in FIFO queue")
    water_level_normalized: float
    water_level_category: str = Field(..., description="Dynamic water level category from fuzzy membership")
    avg_rate_normalized: float
    avg_rate_category: str = Field(..., description="Dynamic rate category from fuzzy membership")
    current_rainfall: float
    rainfall_normalized: float
    current_rainfall_category: str = Field(..., description="Dynamic rainfall category from fuzzy membership")
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

class BatchReadingResult(BaseModel):
    index: int
    reading_number: int
    current_distance: float
    water_depth_from_ground: float
    water_level_category: str
    avg_rate_change_cm_per_min: float
    avg_rate_category: str = Field(..., description="Dynamic rate category")
    rainfall_category: str
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
    reading_interval_seconds: int
    previous_warning_level: Optional[str] = Field(None, description="Last recorded warning level")
    readings_count: int = Field(..., description="Number of readings in FIFO queue")
    history_size: int = Field(60, description="Maximum FIFO queue capacity")

class ThresholdsResponse(BaseModel):
    calibration_height: float
    banjir_level: float
    siaga_level: float
    distance_range: Dict[str, str]
    notification_intervals: Dict[str, Optional[str]]

class SimulationRequest(BaseModel):
    scenario: str = Field(..., description="Scenario type: rising, falling, or stable", example="rising")
    num_readings: int = Field(default=20, description="Number of readings to generate", example=20, gt=0, le=100)
    start_distance: float = Field(default=200.0, description="Starting distance in cm", example=200.0)
    rainfall: float = Field(default=12.5, description="Rainfall intensity (mm/hour)", example=12.5)

class SimulationResult(BaseModel):
    reading_index: int
    reading_number: int
    current_distance: float
    water_depth: float
    water_level_category: str
    avg_rate_change: float
    avg_rate_category: str
    rainfall_category: str
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
        "message": "Flood Early Warning Fuzzy Logic API v4.0",
        "description": "3-input fuzzy with FIFO queue and dynamic categorization",
        "features": [
            "FIFO queue (up to 60 readings) for rate of change calculation",
            "Dynamic fuzzy categorization (adapts to membership function changes)",
            "No timestamps required - automatic FIFO management",
            "Time-to-flood estimation", 
            "Real-time risk assessment",
            "Recovery detection (SIAGA/BANJIR → NORMAL)",
            "Automatic recovery notifications",
            "Override support for siaga and banjir levels",
            "Configurable reading interval"
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
    Get current system status including calibration, warning levels, and FIFO queue status
    """
    return {
        "is_calibrated": fuzzy_system.calibration_height is not None,
        "calibration_height": fuzzy_system.calibration_height,
        "siaga_level": fuzzy_system.siaga_level,
        "banjir_level": fuzzy_system.banjir_level,
        "reading_interval_seconds": fuzzy_system.reading_interval_seconds,
        "previous_warning_level": fuzzy_system.previous_warning_level,
        "readings_count": len(fuzzy_system.distance_history),
        "history_size": 60
    }

@app.post("/api/calibrate", response_model=CalibrationResponse, tags=["System"])
async def calibrate_system(request: CalibrationRequest):
    """
    Calibrate the flood warning system with ground distance measurement
    
    This sets the baseline for all future measurements:
    - BANJIR level = ground_distance (or override with banjir_level)
    - SIAGA level = ground_distance + 30cm (or override with siaga_level)
    - Reading interval = time between readings (default: 1 second)
    
    **Override Support:**
    You can customize the system:
    - Provide `siaga_level` to set a custom warning threshold
    - Provide `banjir_level` to set a custom flood threshold
    - Provide `reading_interval_seconds` to match your sensor's reading frequency
    - If not provided, defaults are used
    
    **Dynamic Categorization:**
    The system now uses fuzzy membership functions to dynamically categorize:
    - Water levels (NORMAL, SIAGA, BANJIR RINGAN, BANJIR PARAH)
    - Rate of change (Turun Sangat Cepat → Naik Ekstrem)
    - Rainfall intensity (Tidak Hujan → Hujan Ekstrem)
    
    Categories automatically adapt if you modify membership functions!
    """
    try:
        # Validate override values if provided
        if request.siaga_level is not None and request.banjir_level is not None:
            if request.siaga_level <= request.banjir_level:
                raise HTTPException(
                    status_code=400, 
                    detail="siaga_level must be greater than banjir_level"
                )
        
        is_overridden = request.siaga_level is not None or request.banjir_level is not None
        
        # Reinitialize system with new reading interval if provided
        global fuzzy_system
        if request.reading_interval_seconds != fuzzy_system.reading_interval_seconds:
            fuzzy_system = DynamicFuzzyFloodWarningSystem(
                reading_interval_seconds=request.reading_interval_seconds
            )
        
        fuzzy_system.calibrate(
            ground_distance=request.ground_distance,
            siaga_level_override=request.siaga_level,
            banjir_level_override=request.banjir_level
        )
        
        message = "System calibrated successfully"
        if is_overridden:
            message += " with custom override levels"
        
        return {
            "success": True,
            "message": message,
            "calibration_height": fuzzy_system.calibration_height,
            "siaga_level": fuzzy_system.siaga_level,
            "banjir_level": fuzzy_system.banjir_level,
            "reading_interval_seconds": fuzzy_system.reading_interval_seconds,
            "is_overridden": is_overridden
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")

@app.post("/api/calculate-risk", response_model=RiskCalculationResponse, tags=["Risk Assessment"])
async def calculate_flood_risk(request: RiskCalculationRequest):
    """
    Calculate flood risk using FIFO queue for rate of change calculation
    
    **Inputs:**
    1. Water Level (current_distance): cm from sensor to water surface
    2. Average Rate of Change: Automatically calculated from FIFO queue (cm/min)
       - Negative = water rising (distance decreasing)
       - Positive = water falling (distance increasing)
    3. Current Rainfall (current_rainfall_mm_per_hour): mm/hour (0-25)
    
    **How FIFO Queue Works:**
    - System maintains up to 60 readings in a FIFO queue
    - When you submit a new reading, it's added to the queue
    - Oldest reading is automatically removed if queue is full
    - Rate of change calculated from first to last reading in queue
    - No timestamps needed - system uses configured reading interval
    
    **Dynamic Categorization:**
    All categories are now determined dynamically by finding the fuzzy set
    with the highest membership degree:
    - Water Level Category: NORMAL, SIAGA, BANJIR RINGAN, or BANJIR PARAH
    - Rate Category: 8 levels from "Turun Sangat Cepat" to "Naik Ekstrem"
    - Rainfall Category: 6 levels from "Tidak Hujan" to "Hujan Ekstrem"
    
    **Outputs:**
    - Risk assessment with warning level (NORMAL, SIAGA, BANJIR)
    - Dynamic categories based on actual fuzzy membership functions
    - Average rate of change over available readings
    - Number of readings used for averaging
    - Risk score and categorized risk level
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
            current_rainfall_mm_per_hour=request.current_rainfall_mm_per_hour
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
    
    **FIFO Queue Behavior:**
    - Readings are processed in order
    - Each reading is added to the FIFO queue
    - Queue automatically maintains only the last 60 readings
    - Rate of change calculated based on available history
    
    **Dynamic Categories:**
    All categories in the response are dynamically determined from fuzzy membership functions.
    """
    if fuzzy_system.calibration_height is None:
        raise HTTPException(status_code=400, detail="System not calibrated. Call /api/calibrate first")
    
    if not request.readings or len(request.readings) == 0:
        raise HTTPException(status_code=400, detail="readings array cannot be empty")
    
    try:
        results = []
        
        for idx, reading in enumerate(request.readings):
            # Calculate flood risk
            result = fuzzy_system.calculate_risk(
                current_distance=reading.current_distance,
                current_rainfall_mm_per_hour=reading.current_rainfall_mm_per_hour
            )
            
            results.append(BatchReadingResult(
                index=idx,
                reading_number=result['reading_number'],
                current_distance=result['current_distance'],
                water_depth_from_ground=round(result['water_depth_from_ground'], 2),
                water_level_category=result['water_level_category'],
                avg_rate_change_cm_per_min=round(result['avg_rate_change_cm_per_min'], 4),
                avg_rate_category=result['avg_rate_category'],
                rainfall_category=result['current_rainfall_category'],
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
            "SIAGA": "10 minutes (risk ≥75%)",
            "BANJIR": "5 minutes (risk ≥75%)"
        }
    }

@app.post("/api/reset", tags=["System"])
async def reset_system():
    """
    Reset the FIFO queue and warning state
    
    This maintains the calibration but clears:
    - All distance readings in FIFO queue
    - Reading count
    - Previous warning level
    
    Use this when you want to start fresh monitoring without recalibrating.
    """
    if fuzzy_system.calibration_height is None:
        raise HTTPException(status_code=400, detail="System not calibrated")
    
    try:
        fuzzy_system.reset_history()
        
        return {
            "success": True, 
            "message": "System history reset successfully. Calibration maintained.",
            "calibration_maintained": True,
            "queue_cleared": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.post("/api/simulate", response_model=SimulationResponse, tags=["Simulation"])
async def simulate_scenario(request: SimulationRequest):
    """
    Simulate a flood scenario with generated data
    
    **Scenarios:**
    - `rising`: Water level rising (distance decreasing by 0.3 cm per reading)
    - `falling`: Water level falling (distance increasing by 0.3 cm per reading)
    - `stable`: Water level stable with small random variations (±0.1 cm)
    
    **Parameters:**
    - `num_readings`: Number of readings to generate (1-100)
    - `start_distance`: Initial distance from sensor to water
    - `rainfall`: Constant rainfall intensity throughout simulation
    
    **FIFO Behavior:**
    - Each reading is added to the FIFO queue in sequence
    - Rate of change is calculated from available history
    - After 60 readings, oldest readings are automatically removed
    
    **Dynamic Categories:**
    All categories shown are dynamically determined from fuzzy membership functions.
    
    This is useful for:
    - Testing the system behavior under different scenarios
    - Training and demonstration purposes
    - Understanding how the FIFO queue works
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
        
        results = []
        
        for i in range(request.num_readings):
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
                current_rainfall_mm_per_hour=request.rainfall
            )
            
            results.append(SimulationResult(
                reading_index=i,
                reading_number=result['reading_number'],
                current_distance=round(result['current_distance'], 2),
                water_depth=round(result['water_depth_from_ground'], 2),
                water_level_category=result['water_level_category'],
                avg_rate_change=round(result['avg_rate_change_cm_per_min'], 4),
                avg_rate_category=result['avg_rate_category'],
                rainfall_category=result['current_rainfall_category'],
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
                "num_readings": request.num_readings,
                "start_distance": request.start_distance,
                "rainfall": request.rainfall,
                "reading_interval_seconds": fuzzy_system.reading_interval_seconds
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
        "api_version": "4.0.0",
        "features": {
            "queue_type": "FIFO (First In, First Out)",
            "queue_size": 60,
            "rate_unit": "cm/min",
            "categorization": "Dynamic (fuzzy membership-based)",
            "timestamp_required": False,
            "override_support": "siaga and banjir levels + reading interval"
        }
    }

if __name__ == "__main__":
    print("=" * 70)
    print("Flood Early Warning Fuzzy Logic API v4.0")
    print("=" * 70)
    print("\n✨ Features:")
    print("  • FIFO queue (up to 60 readings) for stable rate calculation")
    print("  • Dynamic categorization based on fuzzy membership functions")
    print("  • No timestamps required - automatic FIFO management")
    print("  • Time-to-flood estimation")
    print("  • Recovery detection (SIAGA/BANJIR → NORMAL)")
    print("  • Configurable reading interval")
    print("  • Override support for siaga and banjir levels")
    print("\n📡 Endpoints:")
    print("  • POST /api/calibrate       - Calibrate system")
    print("  • POST /api/calculate-risk  - Calculate flood risk")
    print("  • POST /api/batch-readings  - Process multiple readings")
    print("  • GET  /api/status          - Get system status")
    print("  • GET  /api/thresholds      - Get warning thresholds")
    print("  • POST /api/reset           - Reset FIFO queue")
    print("  • POST /api/simulate        - Run simulation")
    print("  • GET  /health              - Health check")
    print("\n🌐 Starting server...")
    print("  • API: http://localhost:8000")
    print("  • Docs: http://localhost:8000/docs")
    print("  • ReDoc: http://localhost:8000/redoc")
    print("=" * 70)
    uvicorn.run("api2:app", host="0.0.0.0", port=8000, reload=True)