from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class BasisType(str, Enum):
    RECTILINEAR = "rectilinear"  # Z-basis: |0⟩, |1⟩
    DIAGONAL = "diagonal"        # X-basis: |+⟩, |-⟩

class QuantumState(BaseModel):
    theta: float
    phi: float
    bloch_x: float
    bloch_y: float
    bloch_z: float

class BB84Result(BaseModel):
    alice_bits: List[int]
    alice_bases: List[int]
    bob_bases: List[int]
    bob_results: List[int]
    alice_key: List[int]
    bob_key: List[int]
    error_rate: float
    key_length: int
    matching_bases: List[bool]
    backend_used: str

class SecurityAnalysis(BaseModel):
    error_rate: float
    threshold: float
    secure: bool
    security_level: str
    recommended_action: str

class EavesdropperResult(BaseModel):
    eve_bases: List[int]
    eve_measurements: List[int]
    interception_rate: float

class EncryptionResult(BaseModel):
    success: bool
    ciphertext: Optional[str] = None
    key_length: Optional[int] = None
    aes_key_length: Optional[int] = None
    iv: Optional[str] = None
    error: Optional[str] = None

class DecryptionResult(BaseModel):
    success: bool
    plaintext: Optional[str] = None
    key_length: Optional[int] = None
    error: Optional[str] = None

class RealTimeMetrics(BaseModel):
    timestamp: float
    key_generation_rate: float
    quantum_error_rate: float
    channel_efficiency: float
    security_parameter: float
    backend_status: str
    queue_length: int
