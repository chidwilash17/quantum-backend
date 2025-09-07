from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .qkd_models import BB84Result, SecurityAnalysis, EavesdropperResult

class QKDResponse(BaseModel):
    message: str
    message_binary: str
    bb84_details: BB84Result
    final_key: List[int]
    key_length: int
    security_analysis: SecurityAnalysis

class EavesdropperSimulationResponse(BaseModel):
    normal_protocol: BB84Result
    eavesdropper_attack: EavesdropperResult
    comparison: Dict[str, float]

class BlochSphereResponse(BaseModel):
    x: float
    y: float
    z: float
    theta: float
    phi: float
    amplitudes: Dict[str, float]

class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
