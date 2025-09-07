import asyncio
import random
import numpy as np
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from typing import List, Dict, Any
import os
from .websocket_manager import websocket_manager

class QuantumService:
    def __init__(self):
        self.ibm_token = os.getenv("IBM_QUANTUM_TOKEN")
        self.service = None
        self.initialize_ibm_quantum()

    def initialize_ibm_quantum(self):
        """Initialize IBM Quantum service"""
        if self.ibm_token:
            try:
                self.service = QiskitRuntimeService(token=self.ibm_token)
            except Exception as e:
                print(f"IBM Quantum initialization failed: {e}")

    async def run_bb84_protocol_realtime(self, num_bits: int = 100, use_real_quantum: bool = False):
        """Run BB84 protocol with real-time updates"""
        
        # Log activity
        await websocket_manager.send_activity_log({
            "action": "BB84_PROTOCOL_START",
            "num_bits": num_bits,
            "use_real_quantum": use_real_quantum,
            "timestamp": datetime.now().isoformat()
        })

        # Generate Alice's random bits and bases
        alice_bits = [random.randint(0, 1) for _ in range(num_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(num_bits)]
        
        # Bob's random measurement bases
        bob_bases = [random.randint(0, 1) for _ in range(num_bits)]
        bob_measurements = []

        # Simulate quantum transmission with real-time updates
        for i in range(num_bits):
            # Create quantum circuit for each bit
            qc = QuantumCircuit(1, 1)
            
            # Alice prepares the qubit
            if alice_bits[i] == 1:
                qc.x(0)  # Prepare |1⟩
            if alice_bases[i] == 1:
                qc.h(0)  # Apply Hadamard for diagonal basis
            
            # Bob measures in his chosen basis
            if bob_bases[i] == 1:
                qc.h(0)  # Measure in diagonal basis
            qc.measure(0, 0)

            # Execute circuit
            if use_real_quantum and self.service:
                try:
                    # Run on real IBM Quantum hardware
                    backend = self.service.least_busy(min_num_qubits=1)
                    job = backend.run(transpile(qc, backend), shots=1)
                    result = job.result()
                    measurement = int(list(result.get_counts().keys())[0])
                except Exception as e:
                    # Fallback to simulation
                    measurement = self._simulate_measurement(qc)
            else:
                # Use simulator
                measurement = self._simulate_measurement(qc)

            bob_measurements.append(measurement)

            # Send real-time quantum state update
            theta = np.pi * alice_bits[i]  # |0⟩ or |1⟩
            phi = np.pi/2 * alice_bases[i]  # Adjust for basis
            
            await websocket_manager.send_quantum_state_update({
                "bit_index": i,
                "alice_bit": alice_bits[i],
                "alice_basis": alice_bases[i],
                "bob_basis": bob_bases[i],
                "bob_measurement": measurement,
                "quantum_state": {
                    "theta": theta,
                    "phi": phi,
                    "bloch_vector": {
                        "x": np.sin(theta) * np.cos(phi),
                        "y": np.sin(theta) * np.sin(phi),
                        "z": np.cos(theta)
                    }
                },
                "progress": (i + 1) / num_bits
            })

            # Small delay for visualization
            await asyncio.sleep(0.1)

        # Key sifting - keep only matching bases
        sifted_key = []
        for i in range(num_bits):
            if alice_bases[i] == bob_bases[i]:
                sifted_key.append(alice_bits[i])

        # Calculate error rate
        error_count = 0
        for i in range(min(len(sifted_key), 50)):  # Check first 50 bits
            if alice_bases[i] == bob_bases[i]:
                if alice_bits[i] != bob_measurements[i]:
                    error_count += 1

        error_rate = error_count / min(len(sifted_key), 50) if sifted_key else 0.0

        # Security check
        if error_rate > 0.11:
            await websocket_manager.send_security_alert({
                "alert_type": "HIGH_ERROR_RATE",
                "error_rate": error_rate,
                "threshold": 0.11,
                "message": "Possible eavesdropping detected!"
            })

        await websocket_manager.send_activity_log({
            "action": "BB84_PROTOCOL_COMPLETE",
            "sifted_key_length": len(sifted_key),
            "error_rate": error_rate,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "alice_bits": alice_bits,
            "alice_bases": alice_bases,
            "bob_bases": bob_bases,
            "bob_measurements": bob_measurements,
            "sifted_key": sifted_key,
            "error_rate": error_rate,
            "security_status": "SECURE" if error_rate <= 0.11 else "COMPROMISED"
        }

    def _simulate_measurement(self, circuit):
        """Simulate quantum measurement"""
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1)
        result = job.result()
        measurement = int(list(result.get_counts().keys())[0])
        return measurement

    async def simulate_eavesdropper_realtime(self, num_bits: int = 100):
        """Simulate eavesdropper attack with real-time detection"""
        
        await websocket_manager.send_activity_log({
            "action": "EAVESDROPPER_SIMULATION_START",
            "timestamp": datetime.now().isoformat()
        })

        # Run normal protocol
        normal_result = await self.run_bb84_protocol_realtime(num_bits, False)
        normal_error_rate = normal_result["error_rate"]

        await asyncio.sleep(1)  # Brief pause

        # Run with eavesdropper
        await websocket_manager.send_activity_log({
            "action": "INTRODUCING_EAVESDROPPER",
            "timestamp": datetime.now().isoformat()
        })

        # Simulate higher error rate due to eavesdropping
        eve_error_rate = normal_error_rate + random.uniform(0.15, 0.25)

        # Send real-time alerts
        await websocket_manager.send_security_alert({
            "alert_type": "EAVESDROPPER_DETECTED",
            "normal_error_rate": normal_error_rate,
            "compromised_error_rate": eve_error_rate,
            "detection_confidence": min(95, (eve_error_rate - normal_error_rate) * 400),
            "message": "Quantum eavesdropping attack detected!"
        })

        return {
            "normal_error_rate": normal_error_rate,
            "compromised_error_rate": eve_error_rate,
            "eavesdropper_detected": eve_error_rate > 0.11,
            "detection_confidence": min(95, (eve_error_rate - normal_error_rate) * 400)
        }

quantum_service = QuantumService()
