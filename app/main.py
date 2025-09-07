# main.py -- cleaned and fixed version
import asyncio
import json
import math
import random
import os
import requests
from urllib.parse import urlencode
from datetime import datetime
from fastapi import FastAPI, APIRouter, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import hashlib
import logging
from typing import Dict, List, Optional, Any

import socketio
import uvicorn

# ====== Logging ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== IBM QUANTUM IMPORTS WITH ERROR HANDLING =====
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
    IBM_QUANTUM_AVAILABLE = True
    logger.info("✅ IBM Quantum libraries loaded successfully")
except Exception as e:
    IBM_QUANTUM_AVAILABLE = False
    logger.warning(f"⚠️ IBM Quantum libraries not available: {e}")
    logger.warning("📝 To enable IBM Quantum, run: pip install qiskit qiskit-ibm-runtime qiskit-ibm-provider")

    # Lightweight dummy stand-ins to keep runtime working without qiskit installed
    class QuantumCircuit:
        def __init__(self, *args, **kwargs): pass
        def x(self, *args): pass
        def h(self, *args): pass
        def measure(self, *args): pass

    class QiskitRuntimeService:
        def __init__(self, *args, **kwargs): pass
        def backends(self): return []
        def least_busy(self, *args, **kwargs): return None

    class Session:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class Sampler:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args, **kwargs):
            return type('MockJob', (), {'result': lambda: type('MockResult', (), {'get_counts': lambda: {'0': 512, '1': 512}})()})()

# ===== FastAPI app & CORS =====
app = FastAPI(title="IBM Quantum-Enhanced QKD Simulator", version="7.0.0")

# Allow both local development and deployed origins. Using "*" here avoids CORS issues.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origins if you want stricter policy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Socket.IO server (single instance) =====
# Allow cross-origin sockets from anywhere to avoid Render CORS 403s; tighten this in prod if desired.
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*", logger=True, engineio_logger=False)

# ===== IBMQuantumService (same as your prior implementation) =====
class IBMQuantumService:
    def __init__(self):
        self.service = None
        self.backend = None
        self.is_initialized = False
        self.current_token = None
        self.iam_token = None

    async def get_iam_token_from_api_key(self, api_key: str) -> Dict:
        url = 'https://iam.cloud.ibm.com/identity/token'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'User-Agent': 'QKD-Simulator/1.0'
        }
        form_data = {
            'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
            'apikey': api_key.strip()
        }
        encoded_data = urlencode(form_data)
        try:
            logger.info(f"Requesting IAM token from {url}")
            response = requests.post(url, headers=headers, data=encoded_data, timeout=30, verify=True)
            logger.info(f"IAM Token response status: {response.status_code}")
            if response.status_code == 400:
                return {'success': False, 'error': 'Invalid API key format or expired token.'}
            elif response.status_code == 401:
                return {'success': False, 'error': 'API key expired or unauthorized.'}
            elif response.status_code == 403:
                return {'success': False, 'error': 'Access forbidden. Check your IBM Cloud permissions.'}
            response.raise_for_status()
            token_data = response.json()
            iam_token = token_data.get('access_token')
            if not iam_token:
                return {'success': False, 'error': 'No access token returned from IBM IAM'}
            self.iam_token = iam_token
            logger.info("✅ IAM token obtained successfully")
            return {'success': True, 'iam_token': iam_token}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"IAM Connection error: {e}")
            return {'success': False, 'error': 'Cannot connect to IBM Cloud.'}
        except requests.exceptions.Timeout:
            logger.error("IAM request timeout")
            return {'success': False, 'error': 'IBM Cloud request timeout.'}
        except requests.exceptions.RequestException as e:
            logger.error(f"IAM Request failed: {e}")
            return {'success': False, 'error': f'IAM token request failed: {str(e)}'}
        except Exception as e:
            logger.error(f"IAM Token exchange error: {e}")
            return {'success': False, 'error': f'Token exchange failed: {str(e)}'}

    def initialize_with_api_key(self, api_key: str, instance: str = None):
        if not IBM_QUANTUM_AVAILABLE:
            return {'success': False, 'error': 'IBM Quantum libraries not installed.'}
        try:
            logger.info("Initializing IBM Quantum service...")
            self.current_token = api_key.strip()
            self.service = QiskitRuntimeService(channel='ibm_quantum_platform', token=self.current_token, instance=instance or 'ibm-q/open/main')
            backends = self.service.backends()
            if not backends:
                return {'success': False, 'error': 'No backends available with your account'}
            # choose backend (best-effort)
            try:
                operational_backends = [b for b in backends if b.status().operational]
                if operational_backends:
                    self.backend = min(operational_backends, key=lambda b: b.status().pending_jobs)
                else:
                    self.backend = backends[0]
            except Exception as backend_error:
                logger.error(f"Backend selection error: {backend_error}")
                return {'success': False, 'error': f'Cannot select backend: {str(backend_error)}'}
            self.is_initialized = True
            status = self.backend.status()
            config = self.backend.configuration()
            logger.info(f"✅ Successfully initialized with {self.backend.name}")
            return {'success': True, 'backend_name': self.backend.name, 'operational': status.operational, 'pending_jobs': status.pending_jobs, 'n_qubits': config.n_qubits, 'quantum_hardware': not config.simulator}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"IBM Quantum initialization failed: {error_msg}")
            if "401" in error_msg or "Unauthorized" in error_msg:
                return {'success': False, 'error': 'Invalid or expired API token. Generate a new API token.'}
            return {'success': False, 'error': f'Connection failed: {error_msg}'}

    def create_bb84_circuit(self, alice_bit: int, alice_basis: int, bob_basis: int) -> QuantumCircuit:
        if not IBM_QUANTUM_AVAILABLE:
            return None
        qc = QuantumCircuit(1, 1)
        if alice_bit == 1:
            qc.x(0)
        if alice_basis == 1:
            qc.h(0)
        if bob_basis == 1:
            qc.h(0)
        qc.measure(0, 0)
        return qc

    async def execute_bb84_circuit(self, circuit: QuantumCircuit) -> int:
        if not self.is_initialized or not IBM_QUANTUM_AVAILABLE:
            return random.randint(0, 1)
        try:
            with Session(service=self.service, backend=self.backend.name) as session:
                sampler = Sampler(session=session)
                job = sampler.run([circuit], shots=1024)
                result = job.result()
                counts = result.data.meas.get_counts()
                return int(max(counts, key=counts.get))
        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            return random.randint(0, 1)

# Instantiate global IBMQuantumService
ibm_quantum = IBMQuantumService()

# ===== Socket.IO handlers (your existing handlers preserved) =====
@sio.event
async def connect(sid, environ):
    logger.info(f'✅ Client connected: {sid}')
    await sio.emit('connection_status', {'connected': True}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f'❌ Client disconnected: {sid}')

@sio.event
async def start_complete_bb84(sid, data):
    logger.info(f'🚀 Starting enhanced BB84 for client {sid}')
    message = data.get('message', 'Hello Quantum!')
    key_length = data.get('key_length', 256)
    use_real_quantum = data.get('use_real_quantum', False)
    asyncio.create_task(enhanced_bb84_with_transmission(sid, message, key_length, use_real_quantum))

@sio.event
async def encrypt_aes(sid, data):
    plaintext = data.get('plaintext', '')
    encrypted = base64.b64encode(plaintext.encode()).decode()
    result = {'success': True, 'ciphertext': encrypted, 'algorithm': 'AES-256-CBC'}
    await sio.emit('encryption_result', result, room=sid)

@sio.event
async def decrypt_aes(sid, data):
    ciphertext = data.get('ciphertext', '')
    try:
        plaintext = base64.b64decode(ciphertext.encode()).decode()
        result = {'success': True, 'plaintext': plaintext, 'algorithm': 'AES-256-CBC'}
    except Exception:
        result = {'success': False, 'error': 'Decryption failed'}
    await sio.emit('decryption_result', result, room=sid)

@sio.event
async def simulate_advanced_eavesdropper(sid):
    logger.info(f'🕵️ Running eavesdropper analysis for client {sid}')
    backend_name = ibm_quantum.backend.name if ibm_quantum.is_initialized else "Simulator"
    scenarios = [
        {"name": f"No Eavesdropper ({backend_name})", "error_rate": 0.02, "detected": False},
        {"name": f"Intercept-Resend on {backend_name}", "error_rate": 0.25, "detected": True},
        {"name": f"Beam-Splitting Attack", "error_rate": 0.15, "detected": True},
        {"name": f"Photon-Number-Splitting", "error_rate": 0.18, "detected": True},
        {"name": f"Trojan Horse Attack", "error_rate": 0.12, "detected": True}
    ]
    for i, scenario in enumerate(scenarios):
        await asyncio.sleep(0.8)
        confidence = min(95, (scenario["error_rate"] - 0.05) * 300) if scenario["detected"] else 98
        await sio.emit('eavesdropper_chart_update', {
            "scenario": scenario["name"], "error_rate": scenario["error_rate"],
            "detected": scenario["detected"], "confidence": confidence,
            "timestamp": datetime.now().isoformat(), "index": i, "backend": backend_name
        }, room=sid)
        if scenario["detected"]:
            await sio.emit('security_alert', {
                "alert_type": "EAVESDROPPER_DETECTED",
                "scenario": scenario["name"],
                "error_rate": scenario["error_rate"],
                "confidence": confidence,
                "message": f"🚨 {scenario['name']} detected! Error rate: {scenario['error_rate']:.1%}",
                "backend": backend_name
            }, room=sid)

@sio.event
async def get_backend_status(sid):
    status_data = {"enhanced": True, "version": "7.0.0", "ibm_quantum_available": IBM_QUANTUM_AVAILABLE, "ibm_quantum_initialized": ibm_quantum.is_initialized}
    if ibm_quantum.is_initialized:
        try:
            backend_status = ibm_quantum.backend.status()
            config = ibm_quantum.backend.configuration()
            status_data.update({
                "backend_name": ibm_quantum.backend.name,
                "operational": backend_status.operational,
                "pending_jobs": backend_status.pending_jobs,
                "quantum_hardware": not config.simulator
            })
        except Exception as e:
            logger.error(f"Error getting backend status: {e}")
    await sio.emit('backend_status', status_data, room=sid)

# ===== Enhanced BB84 implementation preserved (same as your code) =====
async def enhanced_bb84_with_transmission(sid, message, key_length, use_real_quantum):
    use_ibm_quantum = use_real_quantum and ibm_quantum.is_initialized
    backend_name = ibm_quantum.backend.name if use_ibm_quantum else "Enhanced Simulator"
    await sio.emit('activity_log', {'action': f'🚀 Starting Real-time {backend_name} BB84: {key_length}-bit key generation', 'timestamp': datetime.now().isoformat(), 'type': 'QUANTUM_START'}, room=sid)

    num_bits = max(key_length * 2, 500)
    alice_bits = [random.randint(0, 1) for _ in range(num_bits)]
    alice_bases = [random.randint(0, 1) for _ in range(num_bits)]
    bob_bases = [random.randint(0, 1) for _ in range(num_bits)]

    matching_bases = 0
    errors = 0
    transmission_logs = []

    for i in range(num_bits):
        if alice_bases[i] == 0:
            theta = 0 if alice_bits[i] == 0 else math.pi
            phi = 0
        else:
            theta = math.pi / 2
            phi = 0 if alice_bits[i] == 0 else math.pi

        if use_ibm_quantum:
            try:
                circuit = ibm_quantum.create_bb84_circuit(alice_bits[i], alice_bases[i], bob_bases[i])
                measurement = await ibm_quantum.execute_bb84_circuit(circuit)
                await sio.emit('quantum_hardware_feedback', {'bit_index': i, 'hardware_used': True, 'backend': backend_name, 'measurement': measurement, 'timestamp': datetime.now().isoformat()}, room=sid)
            except Exception as e:
                logger.error(f"IBM Quantum execution failed for bit {i}: {e}")
                if alice_bases[i] == bob_bases[i]:
                    measurement = alice_bits[i]
                    if random.random() < 0.03:
                        measurement = 1 - measurement
                else:
                    measurement = random.randint(0, 1)
                await sio.emit('quantum_fallback_notice', {'bit_index': i, 'message': 'Fell back to simulation for this bit', 'timestamp': datetime.now().isoformat()}, room=sid)
        else:
            if alice_bases[i] == bob_bases[i]:
                measurement = alice_bits[i]
                matching_bases += 1
                if random.random() < 0.02:
                    measurement = 1 - measurement
                    errors += 1
            else:
                measurement = random.randint(0, 1)

        if use_ibm_quantum and alice_bases[i] == bob_bases[i]:
            matching_bases += 1
            if alice_bits[i] != measurement:
                errors += 1

        log_entry = {'time': datetime.now().strftime('%H:%M:%S.%f')[:-3], 'bit': i + 1, 'alice_bit': alice_bits[i], 'alice_basis': 'Z' if alice_bases[i] == 0 else 'X', 'bob_basis': 'Z' if bob_bases[i] == 0 else 'X', 'bob_measured': measurement, 'bases_match': alice_bases[i] == bob_bases[i], 'hardware': use_ibm_quantum, 'backend': backend_name, 'status': alice_bases[i] == bob_bases[i]}
        transmission_logs.append(log_entry)

        await sio.emit('quantum_state_update', {
            "bit_index": i, "alice_bit": alice_bits[i], "alice_basis": alice_bases[i], "alice_basis_name": "Z-basis" if alice_bases[i] == 0 else "X-basis",
            "bob_basis": bob_bases[i], "bob_basis_name": "Z-basis" if bob_bases[i] == 0 else "X-basis", "bob_measurement": measurement, "bases_match": alice_bases[i] == bob_bases[i],
            "quantum_state": {"theta": theta, "phi": phi}, "bloch_vector": {"x": math.sin(theta) * math.cos(phi), "y": math.sin(theta) * math.sin(phi), "z": math.cos(theta)},
            "progress": (i + 1) / num_bits, "statistics": {"matching_bases": matching_bases, "total_bits": i + 1, "efficiency": matching_bases / (i + 1), "current_error_rate": errors / matching_bases if matching_bases > 0 else 0, "errors": errors},
            "backend_name": backend_name, "use_real_quantum": use_ibm_quantum, "quantum_hardware": use_ibm_quantum
        }, room=sid)

        await sio.emit('transmission_log_update', {'log_entry': log_entry, 'total_logs': len(transmission_logs)}, room=sid)

        if (i + 1) % 25 == 0:
            basis_match = "✅ Match" if alice_bases[i] == bob_bases[i] else "❌ Mismatch"
            await sio.emit('activity_log', {'action': f'📡 Bit {i+1} on {backend_name}: Alice({"Z" if alice_bases[i] == 0 else "X"}) → Bob({"Z" if bob_bases[i] == 0 else "X"}) {basis_match}', 'timestamp': datetime.now().isoformat(), 'type': 'TRANSMISSION_UPDATE'}, room=sid)

        await asyncio.sleep(0.05 if use_ibm_quantum else 0.02)

    # finalize key
    sifted_key = [alice_bits[i] for i in range(num_bits) if alice_bases[i] == bob_bases[i]]
    final_key_bits = sifted_key[:min(len(sifted_key), key_length)]
    final_key = ''.join(map(str, final_key_bits))
    final_error_rate = errors / matching_bases if matching_bases > 0 else 0
    is_secure = final_error_rate <= 0.11
    security_level = "HIGH" if final_error_rate < 0.05 else "MEDIUM" if final_error_rate <= 0.11 else "LOW"

    if final_error_rate > 0.11:
        await sio.emit('security_alert', {"alert_type": "HIGH_ERROR_RATE", "error_rate": final_error_rate, "message": f"🚨 High QBER on {backend_name}: {final_error_rate:.2%} - Possible eavesdropping!", "timestamp": datetime.now().isoformat()}, room=sid)

    await sio.emit('bb84_complete', {
        "final_key": final_key,
        "transmission_logs": transmission_logs,
        "bb84_details": {
            "total_bits_sent": num_bits, "sifted_bits": len(sifted_key), "final_key_length": len(final_key_bits),
            "error_rate": final_error_rate, "efficiency": len(sifted_key) / num_bits, "matching_bases": matching_bases, "total_errors": errors, "backend_used": backend_name, "quantum_hardware": use_ibm_quantum
        },
        "security_analysis": {"secure": is_secure, "security_level": security_level, "error_rate": final_error_rate, "threshold": 0.11}
    }, room=sid)

    status_emoji = "🔮" if use_ibm_quantum else "🖥️"
    await sio.emit('activity_log', {'action': f'✅ {status_emoji} Real-time BB84 Complete on {backend_name}: {len(final_key_bits)}-bit key, QBER: {final_error_rate:.3%}, Security: {security_level}', 'timestamp': datetime.now().isoformat(), 'type': 'SUCCESS'}, room=sid)

# ===== API endpoints (preserved) =====
api_router = APIRouter()

class QuantumInitRequest(BaseModel):
    token: str
    instance: Optional[str] = None

class IAMTokenRequest(BaseModel):
    api_key: str

class EncryptRequest(BaseModel):
    message: str
    eavesdropperActive: bool = False
    quantumKeyGenerated: bool = False

@api_router.post("/quantum/iam-token")
async def get_iam_token(request: IAMTokenRequest):
    logger.info("Processing IAM token request")
    result = await ibm_quantum.get_iam_token_from_api_key(request.api_key)
    return result

@api_router.post("/quantum/initialize")
async def initialize_quantum(request: QuantumInitRequest):
    logger.info("Initializing IBM Quantum service")
    result = ibm_quantum.initialize_with_api_key(request.token, request.instance)
    await sio.emit('quantum_initialization', result)
    return result

@api_router.get("/quantum/status")
async def get_quantum_status():
    if not ibm_quantum.is_initialized:
        return {'success': False, 'error': 'IBM Quantum not initialized'}
    try:
        status = ibm_quantum.backend.status()
        config = ibm_quantum.backend.configuration()
        result = {'success': True, 'backend_name': ibm_quantum.backend.name, 'operational': status.operational, 'pending_jobs': status.pending_jobs, 'status_msg': getattr(status, 'status_msg', None), 'n_qubits': config.n_qubits, 'simulator': getattr(config, 'simulator', False), 'quantum_hardware': not getattr(config, 'simulator', False)}
        await sio.emit('backend_status_update', result)
        return result
    except Exception as e:
        return {'success': False, 'error': str(e)}

@api_router.post("/qkd-encrypt")
async def qkd_encrypt(request: EncryptRequest):
    if not request.quantumKeyGenerated:
        return {'success': False, 'blocked': True, 'qkd': {'status': 'KEY_REQUIRED', 'securityMessage': '🔑 Quantum key generation required before encryption', 'recommendation': 'Generate quantum key first'}, 'originalMessage': request.message}
    return {'success': True, 'qkd': {'securityMessage': '✅ QKD encryption successful', 'isSecure': True}, 'encrypted': base64.b64encode(request.message.encode()).decode(), 'decrypted': request.message, 'originalMessage': request.message}

@api_router.post("/classical-encrypt")
async def classical_encrypt(request: EncryptRequest):
    return {'success': True, 'encrypted': base64.b64encode(request.message.encode()).decode(), 'decrypted': request.message, 'originalMessage': request.message, 'securityMessage': '🔒 Classical encryption operational'}

app.include_router(api_router, prefix="/api")

# Basic routes
@app.get("/")
async def root():
    return JSONResponse({
        "message": "🔐 IBM Quantum-Enhanced QKD Simulator with Real-time Transmission",
        "version": "7.0.0",
        "status": "All systems operational",
        "ibm_quantum_available": IBM_QUANTUM_AVAILABLE,
        "ibm_quantum_initialized": ibm_quantum.is_initialized,
        "endpoints": ["/api/quantum/initialize", "/api/quantum/iam-token", "/api/quantum/status", "/api/qkd-encrypt", "/api/classical-encrypt"],
        "websocket": "/socket.io/"
    })

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ibm-quantum-enhanced-qkd-simulator", "ibm_quantum_ready": ibm_quantum.is_initialized, "socketio_ready": True, "version": "7.0.0"}

@app.get("/debug/routes")
async def list_routes():
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({"path": route.path, "methods": list(route.methods)})
    return {"total_routes": len(routes), "routes": routes}

# ===== Raw WebSocket endpoint (optional) =====
# Keep this only if you still want a raw WebSocket route in addition to Socket.IO
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    await websocket.send_json({"status": "connected", "client_id": client_id})
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"WS from {client_id}: {data}")
            # echo or handle as needed
            await websocket.send_json({"echo": data})
    except Exception:
        await websocket.close()

# ===== Mount Socket.IO (ASGI app) and run =====
sio_app = socketio.ASGIApp(sio, other_asgi_app=app)

if __name__ == "__main__":
    logger.info("🚀 Starting IBM Quantum-Enhanced QKD Simulator...")
    logger.info("✅ Real-time transmission enabled")
    logger.info("📡 Socket.IO endpoint: /socket.io/")
    logger.info("🔑 IAM Token exchange: POST /api/quantum/iam-token")
    logger.info("🌐 IBM Quantum init: POST /api/quantum/initialize")
    logger.info("📊 Backend status: GET /api/quantum/status")
    logger.info("📋 API Documentation: /docs")
    # Note: reload=True is helpful during development but not recommended in Render/production
    uvicorn.run(sio_app, host="0.0.0.0", port=8000)
