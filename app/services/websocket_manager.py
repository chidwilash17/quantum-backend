import asyncio
import json
import logging
from typing import Dict, Set
from fastapi import WebSocket
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_data: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_data[websocket] = {"client_id": client_id, "connected_at": datetime.now()}
        logger.info(f"WebSocket client {client_id} connected")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        }, websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            client_id = self.connection_data.get(websocket, {}).get("client_id", "unknown")
            self.active_connections.remove(websocket)
            del self.connection_data[websocket]
            logger.info(f"WebSocket client {client_id} disconnected")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        if websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                self.disconnect(websocket)

    async def broadcast(self, message: dict):
        if self.active_connections:
            message["timestamp"] = datetime.now().isoformat()
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn)

    async def send_quantum_state_update(self, state_data: dict):
        """Send real-time quantum state updates"""
        await self.broadcast({
            "type": "quantum_state_update",
            "data": state_data
        })

    async def send_security_alert(self, alert_data: dict):
        """Send real-time security alerts"""
        await self.broadcast({
            "type": "security_alert",
            "data": alert_data
        })

    async def send_activity_log(self, activity_data: dict):
        """Send activity log updates"""
        await self.broadcast({
            "type": "activity_log",
            "data": activity_data
        })

websocket_manager = WebSocketManager()
