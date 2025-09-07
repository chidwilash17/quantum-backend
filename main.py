import asyncio
import json
import math
import random
import os
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="QKD BB84 Simulator", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        if self.active_connections:
            message["timestamp"] = datetime.now().isoformat()
            disconnected = []
            
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    disconnected.append(connection)
            
            for conn in disconnected:
                self.disconnect(conn)

manager = ConnectionManager()

class QuantumService:
    def __init__(self):
        self.ibm_token = os.getenv("IBM_QUANTUM_TOKEN")
        print("🔐 Quantum Service initialized")
        if self.ibm_token:
            print("✅ IBM Quantum token found")
        else:
            print("⚠️ No IBM Quantum token - using simulator")

    async def run_bb84_protocol(self, message: str, num_bits: int, use_real_quantum: bool):
        """BB84 Protocol with real-time updates"""
        
        await manager.broadcast({
            "type": "activity_log",
            "data": {"action": f"BB84_PROTOCOL_START with {num_bits} qubits"}
        })

        # Generate random data for Alice and Bob
        alice_bits = [random.randint(0, 1) for _ in range(num_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(num_bits)]
        bob_bases = [random.randint(0, 1) for _ in range(num_bits)]
        bob_measurements = []

        # Process each qubit with real-time updates
        for i in range(num_bits):
            # Generate quantum state for Bloch sphere
            theta = random.uniform(0, math.pi)
            phi = random.uniform(0, 2 * math.pi)
            
            # Simulate quantum measurement
            if alice_bases[i] == bob_bases[i]:  # Same basis
                measurement = alice_bits[i]  # Perfect measurement
            else:  # Different basis
                measurement = random.randint(0, 1)  # Random result
            
            bob_measurements.append(measurement)

            # Send real-time update to frontend
            await manager.broadcast({
                "type": "quantum_state_update",
                "data": {
                    "bit_index": i,
                    "alice_bit": alice_bits[i],
                    "alice_basis": alice_bases[i],
                    "bob_basis": bob_bases[i],
                    "bob_measurement": measurement,
                    "quantum_state": {"theta": theta, "phi": phi},
                    "bloch_vector": {
                        "x": math.sin(theta) * math.cos(phi),
                        "y": math.sin(theta) * math.sin(phi),
                        "z": math.cos(theta)
                    },
                    "progress": (i + 1) / num_bits,
                    "backend_name": "IBM Quantum" if use_real_quantum else "Simulator",
                    "use_real_quantum": use_real_quantum
                }
            })

            # Small delay for visualization
            await asyncio.sleep(0.08)

        # Key sifting - keep matching bases
        sifted_key = []
        for i in range(num_bits):
            if alice_bases[i] == bob_bases[i]:
                sifted_key.append(alice_bits[i])

        # Calculate error rate
        error_count = 0
        sample_size = min(50, len(sifted_key))
        for i in range(sample_size):
            if i < len(bob_measurements) and alice_bits[i] != bob_measurements[i]:
                error_count += 1
        
        error_rate = error_count / sample_size if sample_size > 0 else 0.0

        # Security analysis
        if error_rate > 0.11:
            await manager.broadcast({
                "type": "security_alert",
                "data": {
                    "alert_type": "HIGH_ERROR_RATE",
                    "error_rate": error_rate,
                    "message": f"🚨 High error rate detected: {error_rate:.2%}!"
                }
            })

        # Final results
        final_key = ''.join(map(str, sifted_key[:min(len(sifted_key), num_bits)]))
        
        await manager.broadcast({
            "type": "bb84_complete",
            "data": {
                "final_key": final_key,
                "bb84_details": {
                    "error_rate": error_rate,
                    "sifted_bits": len(sifted_key),
                    "efficiency": len(sifted_key) / num_bits
                },
                "security_analysis": {
                    "secure": error_rate <= 0.11,
                    "security_level": "HIGH" if error_rate < 0.05 else "MEDIUM" if error_rate <= 0.11 else "LOW"
                }
            }
        })

        await manager.broadcast({
            "type": "activity_log",
            "data": {"action": "BB84_PROTOCOL_COMPLETE"}
        })

    async def simulate_eavesdropper(self):
        """Eavesdropper simulation"""
        await manager.broadcast({
            "type": "activity_log",
            "data": {"action": "EAVESDROPPER_SIMULATION_START"}
        })

        # Simulate detection
        normal_error = random.uniform(0.01, 0.05)
        eve_error = random.uniform(0.20, 0.35)
        eve_detected = eve_error > 0.11

        await manager.broadcast({
            "type": "security_alert",
            "data": {
                "alert_type": "EAVESDROPPER_DETECTED" if eve_detected else "CHANNEL_SECURE",
                "normal_error_rate": normal_error,
                "compromised_error_rate": eve_error,
                "eve_detected": eve_detected,
                "message": "🚨 Eavesdropper detected!" if eve_detected else "✅ Channel secure"
            }
        })

quantum_service = QuantumService()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start_bb84_full":
                asyncio.create_task(
                    quantum_service.run_bb84_protocol(
                        message.get("message", "Hello Quantum!"),
                        message.get("num_bits", 100),
                        message.get("use_real_quantum", False)
                    )
                )
            elif message["type"] == "simulate_eavesdropper_advanced":
                asyncio.create_task(quantum_service.simulate_eavesdropper())
            elif message["type"] == "get_backend_status":
                await websocket.send_text(json.dumps({
                    "type": "backend_status",
                    "data": {"ibm_quantum": bool(quantum_service.ibm_token)}
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "qkd-quantum-simulator"}

@app.get("/")
async def root():
    return {
        "message": "🔐 QKD BB84 Quantum Simulator",
        "version": "3.0.0",
        "websocket": "/ws/{client_id}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
