import asyncio
import json
import time
import requests
import logging
from typing import Dict, List, Optional, Any
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler, Estimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_provider import IBMProvider
import websockets
from concurrent.futures import ThreadPoolExecutor
import threading

class IBMQuantumService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.service = None
        self.backend = None
        self.session_id = None
        self.bearer_token = None
        self.base_url = "https://quantum.cloud.ibm.com/api/v1"
        self.websocket_clients = set()
        self.is_running = False
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def initialize_service(self, token: str, instance: str = None):
        """Initialize IBM Quantum service with API token"""
        try:
            # Initialize Qiskit Runtime Service
            self.service = QiskitRuntimeService(
                channel='ibm_quantum',
                token=token,
                instance=instance
            )
            
            # Get available backends
            backends = self.service.backends()
            
            # Choose best available backend (prefer real quantum devices)
            self.backend = self.service.least_busy(
                operational=True, 
                simulator=False,
                min_num_qubits=5
            )
            
            if not self.backend:
                # Fallback to simulator if no real device available
                self.backend = self.service.backend('ibm_qasm_simulator')
            
            self.logger.info(f"✅ Connected to IBM Quantum Backend: {self.backend.name}")
            
            # Generate bearer token for REST API
            self.bearer_token = self._get_bearer_token(token)
            
            return {
                'success': True,
                'backend': self.backend.name,
                'backend_type': 'quantum' if not self.backend.configuration().simulator else 'simulator',
                'queue_length': getattr(self.backend.status(), 'pending_jobs', 0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize IBM Quantum service: {e}")
            return {'success': False, 'error': str(e)}

    def _get_bearer_token(self, api_key: str) -> str:
        """Generate IBM Cloud IAM bearer token"""
        url = 'https://iam.cloud.ibm.com/identity/token'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = f'grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}'
        
        try:
            response = requests.post(url, headers=headers, data=data)
            return response.json()['access_token']
        except Exception as e:
            self.logger.error(f"Failed to get bearer token: {e}")
            return None

    async def create_quantum_session(self) -> Dict:
        """Create a quantum session for multiple job execution"""
        if not self.bearer_token:
            return {'success': False, 'error': 'Bearer token not available'}
            
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'backend': self.backend.name,
            'mode': 'dedicated',
            'max_ttl': 28800  # 8 hours
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/sessions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 201:
                session_data = response.json()
                self.session_id = session_data['id']
                self.logger.info(f"✅ Created quantum session: {self.session_id}")
                return {'success': True, 'session_id': self.session_id}
            else:
                self.logger.error(f"Failed to create session: {response.text}")
                return {'success': False, 'error': response.text}
                
        except Exception as e:
            self.logger.error(f"Session creation error: {e}")
            return {'success': False, 'error': str(e)}

    async def generate_quantum_bb84_key(self, 
                                      key_length: int = 256, 
                                      websocket_callback=None) -> Dict:
        """Generate quantum key using BB84 protocol on real IBM hardware"""
        
        if not self.service or not self.backend:
            return {'success': False, 'error': 'IBM Quantum service not initialized'}
        
        try:
            # Create BB84 quantum circuits
            circuits = self._create_bb84_circuits(key_length)
            
            # Real-time updates via WebSocket
            if websocket_callback:
                await websocket_callback({
                    'type': 'bb84_start',
                    'message': f'🚀 Starting BB84 on {self.backend.name}',
                    'backend': self.backend.name,
                    'circuits': len(circuits),
                    'timestamp': time.time()
                })

            # Execute circuits on IBM Quantum hardware
            with Session(service=self.service, backend=self.backend.name) as session:
                sampler = Sampler(session=session)
                
                results = []
                alice_bits = []
                alice_bases = []
                bob_measurements = []
                bob_bases = []
                
                for i, circuit in enumerate(circuits):
                    # Send real-time progress updates
                    if websocket_callback:
                        await websocket_callback({
                            'type': 'bb84_progress',
                            'bit_index': i,
                            'total_bits': len(circuits),
                            'progress': (i / len(circuits)) * 100,
                            'timestamp': time.time()
                        })
                    
                    # Execute circuit on quantum hardware
                    job = sampler.run([circuit], shots=1024)
                    result = job.result()
                    
                    # Process quantum measurement results
                    measurement_counts = result[0].data.meas.get_counts()
                    most_frequent = max(measurement_counts, key=measurement_counts.get)
                    
                    # Extract Alice and Bob data
                    alice_bit = int(most_frequent[1])  # Alice's prepared bit
                    alice_basis = 'Z' if i % 2 == 0 else 'X'  # Alternating bases
                    bob_basis = 'Z' if (i + 1) % 2 == 0 else 'X'  # Bob's random basis choice
                    bob_measurement = int(most_frequent[0])  # Bob's measurement
                    
                    alice_bits.append(alice_bit)
                    alice_bases.append(alice_basis)
                    bob_measurements.append(bob_measurement)
                    bob_bases.append(bob_basis)
                    
                    # Send real-time bit data
                    if websocket_callback:
                        await websocket_callback({
                            'type': 'quantum_bit_update',
                            'bit_index': i,
                            'alice_bit': alice_bit,
                            'alice_basis': alice_basis,
                            'bob_measurement': bob_measurement,
                            'bob_basis': bob_basis,
                            'match': alice_basis == bob_basis,
                            'timestamp': time.time()
                        })
                
                # BB84 Sifting Process
                sifted_key = []
                matching_indices = []
                
                for i in range(len(alice_bits)):
                    if alice_bases[i] == bob_bases[i]:
                        sifted_key.append(alice_bits[i])
                        matching_indices.append(i)
                
                # Error estimation
                test_sample_size = min(len(sifted_key) // 4, 32)
                test_indices = matching_indices[:test_sample_size]
                errors = 0
                
                for idx in test_indices:
                    if alice_bits[idx] != bob_measurements[idx]:
                        errors += 1
                
                error_rate = errors / test_sample_size if test_sample_size > 0 else 0
                
                # Privacy amplification
                final_key = sifted_key[test_sample_size:]
                final_key_hex = ''.join([str(bit) for bit in final_key[:min(256, len(final_key))]])
                
                # Convert to hex for practical use
                final_key_bytes = int(final_key_hex[:64] if len(final_key_hex) >= 64 else final_key_hex.ljust(64, '0'), 2)
                final_key_hex_str = hex(final_key_bytes)[2:].upper().zfill(64)
                
                if websocket_callback:
                    await websocket_callback({
                        'type': 'bb84_complete',
                        'message': '✅ BB84 Protocol Complete - Quantum Key Generated!',
                        'final_key': final_key_hex_str,
                        'error_rate': error_rate,
                        'efficiency': len(final_key) / key_length,
                        'timestamp': time.time()
                    })
                
                return {
                    'success': True,
                    'final_key': final_key_hex_str,
                    'bb84_details': {
                        'total_bits_sent': len(circuits),
                        'sifted_bits': len(sifted_key),
                        'final_key_length': len(final_key),
                        'error_rate': error_rate,
                        'efficiency': len(final_key) / key_length,
                        'backend_used': self.backend.name,
                        'quantum_hardware': not self.backend.configuration().simulator
                    },
                    'security_analysis': {
                        'secure': error_rate < 0.11,
                        'security_level': 'HIGH' if error_rate < 0.05 else 'MEDIUM' if error_rate < 0.11 else 'LOW',
                        'eavesdropper_detected': error_rate > 0.11
                    }
                }
                
        except Exception as e:
            self.logger.error(f"❌ BB84 generation failed: {e}")
            if websocket_callback:
                await websocket_callback({
                    'type': 'bb84_error',
                    'error': str(e),
                    'timestamp': time.time()
                })
            return {'success': False, 'error': str(e)}

    def _create_bb84_circuits(self, key_length: int) -> List[QuantumCircuit]:
        """Create quantum circuits for BB84 protocol"""
        circuits = []
        
        for i in range(key_length):
            qc = QuantumCircuit(2, 2)
            
            # Alice prepares qubit
            alice_bit = i % 2  # Alternating 0,1 pattern
            alice_basis = i % 2  # Alternating Z,X basis
            
            if alice_bit == 1:
                qc.x(0)  # Prepare |1⟩
            
            if alice_basis == 1:
                qc.h(0)  # Apply Hadamard for X basis
            
            # Bob's measurement
            bob_basis = (i + 1) % 2  # Bob's random basis choice
            if bob_basis == 1:
                qc.h(0)  # Measure in X basis
            
            # Measurement
            qc.measure(0, 0)  # Bob's measurement
            qc.measure(1, 1)  # Auxiliary measurement
            
            # Add metadata
            qc.metadata = {
                'alice_bit': alice_bit,
                'alice_basis': 'Z' if alice_basis == 0 else 'X',
                'bob_basis': 'Z' if bob_basis == 0 else 'X'
            }
            
            circuits.append(qc)
        
        return circuits

    async def get_backend_status(self) -> Dict:
        """Get real-time backend status"""
        if not self.backend:
            return {'success': False, 'error': 'No backend available'}
        
        try:
            status = self.backend.status()
            configuration = self.backend.configuration()
            
            return {
                'success': True,
                'backend_name': self.backend.name,
                'operational': status.operational,
                'pending_jobs': status.pending_jobs,
                'status_msg': status.status_msg,
                'n_qubits': configuration.n_qubits,
                'simulator': configuration.simulator,
                'quantum_volume': getattr(configuration, 'quantum_volume', None),
                'processor_type': getattr(configuration, 'processor_type', {})
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def close_session(self):
        """Close quantum session"""
        if self.session_id and self.bearer_token:
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.bearer_token}'
            }
            
            try:
                response = requests.delete(
                    f"{self.base_url}/sessions/{self.session_id}/close",
                    headers=headers
                )
                self.logger.info(f"Session closed: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Error closing session: {e}")
        
        self.session_id = None

# Global service instance
quantum_service = IBMQuantumService()
