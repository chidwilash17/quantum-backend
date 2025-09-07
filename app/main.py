import asyncio
import json
import math
import random
import os
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import socketio

# Load environment variables
load_dotenv()

# Create Socket.IO server with CORS
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[
        "https://quantum-key-distribution-df2l.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000"
    ],
    logger=True,
    engineio_logger=True
)

# Create FastAPI app
app = FastAPI(
    title="QKD BB84 Quantum Simulator", 
    version="3.0.0",
    description="Quantum Key Distribution BB84 Protocol Simulator with Real-time WebSocket Support"
)

# CORS Middleware - CRITICAL: Updated with your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://quantum-key-distribution-df2l.vercel.app",  # Your Vercel frontend
        "http://localhost:3000",
        "http://localhost:5173", 
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def _init_(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.socket_connections: Dict[str, str] = {}

    async def connect_websocket(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"✅ WebSocket Client {client_id} connected. Total: {len(self.active_connections)}")

    def disconnect_websocket(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"❌ WebSocket Client {client_id} disconnected")

    async def broadcast_websocket(self, message: dict):
        if self.active_connections:
            message["timestamp"] = datetime.now().isoformat()
            disconnected = []
            
            for client_id, connection in self.active_connections.items():
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"❌ Failed to send to {client_id}: {e}")
                    disconnected.append(client_id)
            
            for client_id in disconnected:
                self.disconnect_websocket(client_id)

    async def broadcast_socketio(self, event: str, data: dict):
        """Broadcast via Socket.IO"""
        data["timestamp"] = datetime.now().isoformat()
        await sio.emit(event, data)

# Global connection manager
manager = ConnectionManager()

class QuantumService:
    def _init_(self):
        self.ibm_token = os.getenv("IBM_QUANTUM_TOKEN")
        self.backend_status = "ready"
        print("🔐 Quantum Service initialized")
        if self.ibm_token:
            print("✅ IBM Quantum token found")
        else:
            print("⚠ No IBM Quantum token - using simulator")

    async def run_bb84_protocol(self, message: str, num_bits: int, use_real_quantum: bool = False):
        """Enhanced BB84 Protocol with real-time updates"""
        
        # Start notification
        await manager.broadcast_socketio("activity_log", {
            "action": f"🚀 Starting BB84 Protocol: {num_bits} qubits",
            "type": "BB84_START"
        })

        # Generate Alice's random bits and bases
        alice_bits = [random.randint(0, 1) for _ in range(num_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(num_bits)]  # 0=Z, 1=X
        bob_bases = [random.randint(0, 1) for _ in range(num_bits)]
        bob_measurements = []
        transmission_logs = []

        # Process each qubit with real-time updates
        for i in range(num_bits):
            # Generate quantum state parameters for Bloch sphere
            if alice_bases[i] == 0:  # Z basis
                theta = 0 if alice_bits[i] == 0 else math.pi
                phi = 0
            else:  # X basis  
                theta = math.pi / 2
                phi = 0 if alice_bits[i] == 0 else math.pi

            # Add some quantum noise
            theta += random.uniform(-0.1, 0.1)
            phi += random.uniform(-0.1, 0.1)

            # Simulate quantum measurement
            if alice_bases[i] == bob_bases[i]:  # Same basis - perfect correlation
                measurement = alice_bits[i]
                if random.random() < 0.02:  # 2% noise
                    measurement = 1 - measurement
            else:  # Different basis - random result
                measurement = random.randint(0, 1)
            
            bob_measurements.append(measurement)

            # Create transmission log entry
            log_entry = {
                "bit_index": i,
                "alice_bit": alice_bits[i],
                "alice_basis": "Z" if alice_bases[i] == 0 else "X",
                "bob_basis": "Z" if bob_bases[i] == 0 else "X", 
                "bob_measurement": measurement,
                "bases_match": alice_bases[i] == bob_bases[i],
                "timestamp": datetime.now().isoformat()
            }
            transmission_logs.append(log_entry)

            # Calculate current statistics
            processed_bits = i + 1
            matching_bases = sum(1 for j in range(processed_bits) if alice_bases[j] == bob_bases[j])
            current_efficiency = matching_bases / processed_bits if processed_bits > 0 else 0
            
            # Calculate current error rate for matching bases
            errors = 0
            matches = 0
            for j in range(processed_bits):
                if alice_bases[j] == bob_bases[j]:
                    matches += 1
                    if alice_bits[j] != bob_measurements[j]:
                        errors += 1
            current_error_rate = errors / matches if matches > 0 else 0

            # Send real-time quantum state update
            await manager.broadcast_socketio("quantum_state_update", {
                "bit_index": i,
                "alice_bit": alice_bits[i],
                "alice_basis": alice_bases[i],
                "alice_basis_name": "Z-basis" if alice_bases[i] == 0 else "X-basis",
                "bob_basis": bob_bases[i],
                "bob_basis_name": "Z-basis" if bob_bases[i] == 0 else "X-basis",
                "bob_measurement": measurement,
                "bases_match": alice_bases[i] == bob_bases[i],
                "quantum_state": {
                    "theta": theta,
                    "phi": phi,
                    "r": 1.0
                },
                "bloch_vector": {
                    "x": math.sin(theta) * math.cos(phi),
                    "y": math.sin(theta) * math.sin(phi),
                    "z": math.cos(theta)
                },
                "progress": processed_bits / num_bits,
                "statistics": {
                    "processed_bits": processed_bits,
                    "matching_bases": matching_bases,
                    "efficiency": current_efficiency,
                    "current_error_rate": current_error_rate
                },
                "backend_info": {
                    "backend_name": "IBM Quantum Simulator" if use_real_quantum else "Local Simulator",
                    "use_real_quantum": use_real_quantum,
                    "quantum_hardware": use_real_quantum
                }
            })

            # Send transmission log update
            await manager.broadcast_socketio("transmission_log_update", {
                "log_entry": log_entry
            })

            # Small delay for visualization
            await asyncio.sleep(0.05)

        # Key sifting - keep only matching bases
        sifted_key = []
        sifted_indices = []
        for i in range(num_bits):
            if alice_bases[i] == bob_bases[i]:
                sifted_key.append(alice_bits[i])
                sifted_indices.append(i)

        # Calculate final error rate
        error_count = 0
        for i in sifted_indices:
            if alice_bits[i] != bob_measurements[i]:
                error_count += 1
        
        final_error_rate = error_count / len(sifted_key) if len(sifted_key) > 0 else 0.0
        efficiency = len(sifted_key) / num_bits

        # Security analysis
        is_secure = final_error_rate <= 0.11
        if final_error_rate > 0.11:
            await manager.broadcast_socketio("security_alert", {
                "alert_type": "HIGH_ERROR_RATE",
                "error_rate": final_error_rate,
                "threshold": 0.11,
                "message": f"🚨 High error rate detected: {final_error_rate:.2%}! Possible eavesdropping.",
                "secure": False
            })

        # Generate final key
        final_key = ''.join(map(str, sifted_key))
        
        # Complete protocol notification
        await manager.broadcast_socketio("bb84_complete", {
            "success": True,
            "final_key": final_key,
            "message": message,
            "bb84_details": {
                "total_bits": num_bits,
                "sifted_bits": len(sifted_key),
                "final_key_length": len(final_key),
                "error_rate": final_error_rate,
                "efficiency": efficiency,
                "basis_matching_rate": len(sifted_key) / num_bits
            },
            "security_analysis": {
                "secure": is_secure,
                "security_level": "HIGH" if final_error_rate < 0.05 else "MEDIUM" if final_error_rate <= 0.11 else "LOW",
                "qber": final_error_rate,
                "threat_detected": final_error_rate > 0.11
            },
            "transmission_logs": transmission_logs,
            "backend_info": {
                "backend_name": "IBM Quantum" if use_real_quantum else "Simulator",
                "use_real_quantum": use_real_quantum
            }
        })

        await manager.broadcast_socketio("activity_log", {
            "action": f"✅ BB84 Complete: {len(final_key)}-bit key generated",
            "type": "BB84_COMPLETE"
        })

        return {
            "success": True,
            "final_key": final_key,
            "error_rate": final_error_rate,
            "efficiency": efficiency
        }

    async def simulate_advanced_eavesdropper(self):
        """Advanced eavesdropper simulation"""
        await manager.broadcast_socketio("activity_log", {
            "action": "🕵 Starting Advanced Eavesdropper Analysis",
            "type": "EAVESDROPPER_START"
        })

        # Simulate different attack scenarios
        scenarios = [
            {"name": "Intercept-Resend", "error_rate": random.uniform(0.20, 0.30)},
            {"name": "Beam Splitting", "error_rate": random.uniform(0.15, 0.25)},
            {"name": "Photon Number Splitting", "error_rate": random.uniform(0.10, 0.18)},
            {"name": "No Attack", "error_rate": random.uniform(0.01, 0.05)}
        ]

        selected_scenario = random.choice(scenarios)
        eve_detected = selected_scenario["error_rate"] > 0.11

        # Send eavesdropper analysis results
        await manager.broadcast_socketio("security_alert", {
            "alert_type": "EAVESDROPPER_ANALYSIS",
            "scenario": selected_scenario["name"],
            "error_rate": selected_scenario["error_rate"],
            "eve_detected": eve_detected,
            "threat_level": "HIGH" if eve_detected else "NONE",
            "message": f"🚨 {selected_scenario['name']} attack detected!" if eve_detected else f"✅ No eavesdropping detected ({selected_scenario['name']})",
            "recommendations": [
                "Abort key exchange" if eve_detected else "Proceed with key",
                "Investigate channel security" if eve_detected else "Channel is secure"
            ]
        })

        # Send chart data for visualization
        for i in range(10):
            await manager.broadcast_socketio("eavesdropper_chart_update", {
                "measurement": i,
                "normal_error": random.uniform(0.01, 0.05),
                "eve_error": selected_scenario["error_rate"] + random.uniform(-0.02, 0.02),
                "threshold": 0.11
            })
            await asyncio.sleep(0.2)

# Global quantum service instance
quantum_service = QuantumService()

# Socket.IO Event Handlers
@sio.event
async def connect(sid, environ):
    print(f"✅ Socket.IO Client {sid} connected")
    await sio.emit('connected', {
        'message': 'Connected to Quantum Backend',
        'backend': 'FastAPI + Socket.IO',
        'sid': sid
    })

@sio.event
async def disconnect(sid):
    print(f"❌ Socket.IO Client {sid} disconnected")

@sio.event
async def start_complete_bb84(sid, data):
    """Handle BB84 protocol start"""
    print(f"🚀 Starting BB84 for client {sid}: {data}")
    
    message = data.get('message', 'Quantum Key Distribution')
    key_length = data.get('key_length', 100)
    use_real_quantum = data.get('use_real_quantum', False)
    
    # Run BB84 protocol
    asyncio.create_task(
        quantum_service.run_bb84_protocol(message, key_length, use_real_quantum)
    )

@sio.event
async def simulate_advanced_eavesdropper(sid, data):
    """Handle eavesdropper simulation"""
    print(f"🕵 Starting eavesdropper simulation for client {sid}")
    asyncio.create_task(quantum_service.simulate_advanced_eavesdropper())

@sio.event
async def get_backend_status(sid, data):
    """Get backend status"""
    await sio.emit('backend_status', {
        'ibm_quantum_available': bool(quantum_service.ibm_token),
        'backend_status': quantum_service.backend_status,
        'server_time': datetime.now().isoformat()
    })

# FastAPI Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🔐 QKD BB84 Quantum Simulator Backend",
        "version": "3.0.0",
        "status": "running",
        "websocket_endpoint": "/ws/{client_id}",
        "socketio_endpoint": "/socket.io/",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "qkd-quantum-simulator",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "ibm_quantum": bool(quantum_service.ibm_token)
    }

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "api_status": "online",
        "quantum_backend": "ready",
        "websocket_connections": len(manager.active_connections),
        "socketio_available": True
    }

# Legacy WebSocket support (for backward compatibility)
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect_websocket(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "start_bb84_full":
                asyncio.create_task(
                    quantum_service.run_bb84_protocol(
                        message.get("message", "Hello Quantum!"),
                        message.get("num_bits", 100),
                        message.get("use_real_quantum", False)
                    )
                )
            elif message.get("type") == "simulate_eavesdropper_advanced":
                asyncio.create_task(quantum_service.simulate_advanced_eavesdropper())
            elif message.get("type") == "get_backend_status":
                await websocket.send_text(json.dumps({
                    "type": "backend_status",
                    "data": {
                        "ibm_quantum": bool(quantum_service.ibm_token),
                        "status": "ready"
                    }
                }))
                
    except WebSocketDisconnect:
        manager.disconnect_websocket(client_id)
        print(f"❌ WebSocket client {client_id} disconnected")

# Combine FastAPI with Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio, app)

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_endpoints": ["/", "/health", "/docs", "/api/status"]}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(socket_app, host="0.0.0.0", port=port, log_level="info")
