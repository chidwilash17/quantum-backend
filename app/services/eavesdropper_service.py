import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from .quantum_service import QuantumService

class EavesdropperService:
    def __init__(self):
        self.quantum_service = QuantumService()
        self.simulator = AerSimulator()
        
    async def intercept_and_resend(self, alice_bits: List[int], alice_bases: List[int], 
                                  interception_probability: float = 1.0) -> Dict:
        """
        Simulate Eve's intercept-and-resend attack
        """
        eve_bases = np.random.randint(0, 2, len(alice_bits))
        eve_measurements = []
        eve_resent_bits = []
        
        intercepted_qubits = []
        
        for i in range(len(alice_bits)):
            if random.random() < interception_probability:
                # Eve intercepts this qubit
                intercepted_qubits.append(i)
                
                # Eve measures with her chosen basis
                if alice_bases[i] == eve_bases[i]:
                    # Same basis - Eve gets the correct result
                    measurement = alice_bits[i]
                else:
                    # Different basis - 50% chance due to quantum mechanics
                    measurement = random.randint(0, 1)
                
                eve_measurements.append(measurement)
                
                # Eve resends the qubit based on her measurement
                eve_resent_bits.append(measurement)
            else:
                # Qubit passes through without interception
                eve_measurements.append(None)
                eve_resent_bits.append(alice_bits[i])
        
        return {
            "eve_bases": eve_bases.tolist(),
            "eve_measurements": eve_measurements,
            "eve_resent_bits": eve_resent_bits,
            "intercepted_qubits": intercepted_qubits,
            "interception_rate": len(intercepted_qubits) / len(alice_bits),
            "attack_type": "intercept_and_resend"
        }
    
    async def beam_splitting_attack(self, alice_bits: List[int], alice_bases: List[int],
                                   splitting_ratio: float = 0.5) -> Dict:
        """
        Simulate Eve's beam splitting attack
        """
        eve_intercepted = []
        alice_remaining = []
        
        for i in range(len(alice_bits)):
            if random.random() < splitting_ratio:
                # Eve gets a copy of the photon
                eve_intercepted.append({
                    'bit': alice_bits[i],
                    'basis': alice_bases[i],
                    'index': i
                })
                
                # Alice's photon is weakened but still transmitted
                alice_remaining.append({
                    'bit': alice_bits[i],
                    'basis': alice_bases[i],
                    'index': i,
                    'intensity': 1.0 - splitting_ratio
                })
            else:
                # Photon passes through unchanged
                alice_remaining.append({
                    'bit': alice_bits[i],
                    'basis': alice_bases[i],
                    'index': i,
                    'intensity': 1.0
                })
        
        return {
            "eve_intercepted": eve_intercepted,
            "alice_remaining": alice_remaining,
            "splitting_ratio": splitting_ratio,
            "attack_type": "beam_splitting"
        }
    
    def calculate_detection_probability(self, attack_type: str, 
                                      attack_parameters: Dict) -> float:
        """
        Calculate probability of detecting the eavesdropping attack
        """
        if attack_type == "intercept_and_resend":
            # Theoretical detection probability for intercept-and-resend
            interception_rate = attack_parameters.get("interception_rate", 1.0)
            return 1 - (3/4) ** interception_rate
            
        elif attack_type == "beam_splitting":
            splitting_ratio = attack_parameters.get("splitting_ratio", 0.5)
            return splitting_ratio * 0.25
            
        return 0.0
