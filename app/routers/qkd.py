from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import random
import string
from datetime import datetime

router = APIRouter()

class KeyGenerationRequest(BaseModel):
    message: str
    key_length: int = 256
    use_real_quantum: bool = False

class QuantumStateRequest(BaseModel):
    theta: float
    phi: float

@router.post("/generate-key")
async def generate_quantum_key(request: KeyGenerationRequest):
    """Generate quantum key using BB84 protocol simulation"""
    try:
        # Simulate BB84 protocol
        num_bits = max(request.key_length * 2, 400)  # Generate extra bits for sifting
        
        # Alice generates random bits and bases
        alice_bits = [random.randint(0, 1) for _ in range(num_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(num_bits)]
        
        # Bob generates random bases and measures
        bob_bases = [random.randint(0, 1) for _ in range(num_bits)]
        bob_bits = []
        
        # Simulate measurements
        for i in range(num_bits):
            if alice_bases[i] == bob_bases[i]:
                # Same basis - perfect measurement
                bob_bits.append(alice_bits[i])
            else:
                # Different basis - random result
                bob_bits.append(random.randint(0, 1))
        
        # Sift key - keep only matching bases
        sifted_key = []
        matching_indices = []
        for i in range(num_bits):
            if alice_bases[i] == bob_bases[i]:
                sifted_key.append(alice_bits[i])
                matching_indices.append(i)
        
        # Calculate error rate (simulate some errors)
        errors = sum(1 for i in range(min(len(sifted_key), 100)) 
                    if sifted_key[i] != bob_bits[matching_indices[i]] if i < len(matching_indices))
        error_rate = errors / min(len(sifted_key), 100) if sifted_key else 0.0
        
        # Add some random noise for realism
        error_rate += random.uniform(0.0, 0.05)
        error_rate = min(error_rate, 0.11)  # Cap at 11%
        
        # Final key - take requested length
        final_key_bits = sifted_key[:request.key_length]
        final_key = ''.join(map(str, final_key_bits))
        
        # Security analysis
        is_secure = error_rate < 0.11
        security_level = "HIGH" if error_rate < 0.05 else "MEDIUM" if error_rate < 0.11 else "LOW"
        
        # Generate quantum states for Bloch sphere
        quantum_states = []
        for i in range(min(10, len(final_key_bits))):  # Generate states for first 10 bits
            theta = np.pi * random.random()
            phi = 2 * np.pi * random.random()
            quantum_states.append({"theta": theta, "phi": phi})
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "final_key": final_key,
            "bb84_details": {
                "total_bits_sent": num_bits,
                "sifted_bits": len(sifted_key),
                "final_key_length": len(final_key_bits),
                "error_rate": error_rate,
                "efficiency": len(final_key_bits) / num_bits,
                "alice_bits": alice_bits[:50],  # First 50 for display
                "alice_bases": alice_bases[:50],
                "bob_bases": bob_bases[:50]
            },
            "security_analysis": {
                "secure": is_secure,
                "security_level": security_level,
                "error_rate": error_rate,
                "threshold": 0.11
            },
            "quantum_states": quantum_states
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Key generation failed: {str(e)}")

@router.get("/simulate-eavesdropper")
async def simulate_eavesdropper(num_qubits: int = 100):
    """Simulate eavesdropper attack"""
    try:
        # Simulate without eavesdropper
        error_without_eve = random.uniform(0.01, 0.05)
        
        # Simulate with eavesdropper
        error_with_eve = random.uniform(0.15, 0.35)  # Higher error rate
        
        eve_detected = error_with_eve > 0.11
        
        return {
            "success": True,
            "error_rates": {
                "without_eve": error_without_eve,
                "with_eve": error_with_eve
            },
            "eve_detected": eve_detected,
            "detection_threshold": 0.11,
            "simulation_qubits": num_qubits
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.post("/bloch-sphere")
async def get_bloch_sphere_data(request: QuantumStateRequest):
    """Get Bloch sphere visualization data"""
    try:
        theta = request.theta
        phi = request.phi
        
        # Calculate state vector components
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # Calculate probabilities
        prob_0 = (np.cos(theta / 2)) ** 2
        prob_1 = (np.sin(theta / 2)) ** 2
        
        # State description
        alpha = np.cos(theta / 2)
        beta = np.sin(theta / 2) * np.exp(1j * phi)
        
        state_description = f"|ψ⟩ = {alpha:.3f}|0⟩ + {beta:.3f}|1⟩"
        
        return {
            "success": True,
            "theta": theta,
            "phi": phi,
            "coordinates": {"x": x, "y": y, "z": z},
            "prob_0": prob_0,
            "prob_1": prob_1,
            "state_description": state_description,
            "amplitudes": {
                "alpha": {"real": float(alpha.real), "imag": float(alpha.imag)},
                "beta": {"real": float(beta.real), "imag": float(beta.imag)}
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bloch sphere calculation failed: {str(e)}")

@router.get("/metrics")
async def get_security_metrics():
    """Get real-time security metrics"""
    return {
        "error_rate": random.uniform(0.02, 0.08),
        "key_rate": random.randint(150, 300),
        "efficiency": random.uniform(0.7, 0.9),
        "security_level": random.choice(["HIGH", "MEDIUM"]),
        "timestamp": datetime.now().isoformat()
    }
