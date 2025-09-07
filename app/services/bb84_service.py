import numpy as np
from typing import List, Dict, Tuple
import random
from .quantum_service import QuantumService

class BB84Service:
    def __init__(self):
        self.quantum_service = QuantumService()
    
    async def run_complete_bb84(self, message: str, key_length: int = 16):
        """
        Run complete BB84 protocol for QKD
        """
        # Convert message to binary
        message_binary = ''.join(format(ord(c), '08b') for c in message)
        required_key_bits = max(len(message_binary), key_length * 8)
        
        # Run BB84 with enough qubits to generate required key
        num_qubits = required_key_bits * 3  # Account for ~50% key rate and some extra
        
        bb84_result = await self.quantum_service.run_bb84_protocol(num_qubits)
        
        # Extract final key
        final_key = bb84_result["alice_key"][:required_key_bits]
        
        return {
            "message": message,
            "message_binary": message_binary,
            "bb84_details": bb84_result,
            "final_key": final_key,
            "key_length": len(final_key),
            "security_analysis": self.analyze_security(bb84_result["error_rate"])
        }
    
    def analyze_security(self, error_rate: float) -> Dict:
        """Analyze security based on error rate"""
        threshold = 0.11  # Theoretical BB84 threshold
        
        return {
            "error_rate": error_rate,
            "threshold": threshold,
            "secure": error_rate < threshold,
            "security_level": "HIGH" if error_rate < 0.05 else "MEDIUM" if error_rate < threshold else "COMPROMISED",
            "recommended_action": "PROCEED" if error_rate < threshold else "ABORT - POSSIBLE EAVESDROPPING"
        }
    
    async def simulate_with_eavesdropper(self, num_qubits: int = 16):
        """Simulate BB84 with eavesdropper present"""
        # Normal BB84
        normal_result = await self.quantum_service.run_bb84_protocol(num_qubits)
        
        # With eavesdropper
        eve_attack = await self.quantum_service.simulate_eavesdropper(
            normal_result["alice_bits"],
            normal_result["alice_bases"]
        )
        
        # Calculate increased error rate due to eavesdropping
        increased_error_rate = normal_result["error_rate"] + 0.25  # Theoretical increase
        
        return {
            "normal_protocol": normal_result,
            "eavesdropper_attack": eve_attack,
            "comparison": {
                "normal_error_rate": normal_result["error_rate"],
                "compromised_error_rate": increased_error_rate,
                "detection_probability": min(1.0, increased_error_rate / 0.11)
            }
        }
