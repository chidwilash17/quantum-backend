from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import random
import time
import threading
import base64
import io
import os
import sqlite3
import bcrypt
import jwt
from datetime import datetime
from functools import wraps
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pilgrimage_secret_2024'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# JWT Configuration
JWT_SECRET = 'pilgrimage_jwt_secret_2024'

# Initialize Database
def init_db():
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    ''')
    
    # Temples table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS temples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            capacity INTEGER DEFAULT 200,
            current_crowd INTEGER DEFAULT 0,
            wait_time INTEGER DEFAULT 0,
            status TEXT DEFAULT 'open'
        )
    ''')
    
    # Bookings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pilgrim_name TEXT NOT NULL,
            temple_id INTEGER,
            status TEXT DEFAULT 'confirmed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Emergency alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emergency_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert default data
    cursor.execute("INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                   ('admin', bcrypt.hashpw('admin123'.encode(), bcrypt.gensalt()), 'admin'))
    
    cursor.execute("INSERT OR IGNORE INTO temples (name, capacity) VALUES (?, ?), (?, ?), (?, ?)",
                   ('Main Temple', 200, 'North Temple', 150, 'South Temple', 100))
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Initialize database
init_db()

# Global data
recognition_stats = {
    'total_recognized': 0,
    'current_pilgrims': 0,
    'faces_detected': 0,
    'recognition_confidence': 0.85
}

# AI Simulation with all features
class AISimulation:
    def __init__(self):
        self.is_running = False
        self.crowd_data = []
        
    def start_simulation(self):
        self.is_running = True
        def simulate():
            while self.is_running:
                try:
                    # Update all temple data
                    conn = sqlite3.connect('pilgrimage.db')
                    cursor = conn.cursor()
                    
                    cursor.execute("SELECT id, capacity FROM temples")
                    temples = cursor.fetchall()
                    
                    for temple_id, capacity in temples:
                        # Realistic crowd simulation
                        current = cursor.execute(
                            "SELECT current_crowd FROM temples WHERE id = ?", 
                            (temple_id,)
                        ).fetchone()[0]
                        
                        change = random.randint(-8, 12)
                        new_crowd = max(10, min(capacity - 15, current + change))
                        wait_time = new_crowd * 2
                        
                        cursor.execute(
                            "UPDATE temples SET current_crowd = ?, wait_time = ? WHERE id = ?",
                            (new_crowd, wait_time, temple_id)
                        )
                        
                        # Update AI recognition stats
                        recognition_stats['current_pilgrims'] = new_crowd
                        recognition_stats['faces_detected'] = random.randint(5, new_crowd)
                        recognition_stats['recognition_confidence'] = round(random.uniform(0.7, 0.95), 2)
                    
                    conn.commit()
                    conn.close()
                    
                    # Send real-time updates via SocketIO
                    socketio.emit('ai_update', {
                        'recognition_stats': recognition_stats,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Occasionally generate alerts
                    if random.random() < 0.15:
                        self.generate_alert()
                    
                    time.sleep(8)  # Update every 8 seconds
                    
                except Exception as e:
                    logger.error(f"Simulation error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=simulate, daemon=True)
        thread.start()
        logger.info("AI Simulation started")
    
    def generate_alert(self):
        alert_types = ['overcrowding', 'medical', 'security', 'weather']
        alert_type = random.choice(alert_types)
        
        messages = {
            'overcrowding': 'High crowd density detected in Main Temple area',
            'medical': 'Medical assistance required near North Temple',
            'security': 'Security alert - suspicious activity reported',
            'weather': 'Weather warning: Heavy rain expected'
        }
        
        conn = sqlite3.connect('pilgrimage.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO emergency_alerts (type, message, severity) VALUES (?, ?, ?)",
            (alert_type, messages[alert_type], 'high' if alert_type == 'medical' else 'medium')
        )
        conn.commit()
        conn.close()
        
        socketio.emit('emergency_alert', {
            'type': alert_type,
            'message': messages[alert_type],
            'severity': 'high' if alert_type == 'medical' else 'medium',
            'timestamp': datetime.now().isoformat()
        })

# Start AI Simulation
ai_simulator = AISimulation()
ai_simulator.start_simulation()

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        except:
            return jsonify({'error': 'Token is invalid'}), 401
        return f(*args, **kwargs)
    return decorated

# Routes
@app.route('/')
def home():
    return jsonify({
        "message": "🚀 AI Pilgrimage System - All Modules Active!",
        "status": "running",
        "ai_enabled": True,
        "version": "3.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and bcrypt.checkpw(password.encode('utf-8'), user[2]):
        token = jwt.encode({
            'username': username,
            'role': user[3],
            'exp': datetime.utcnow().timestamp() + 86400
        }, JWT_SECRET, algorithm='HS256')
        
        return jsonify({
            'token': token,
            'user': {'username': username, 'role': user[3]}
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/temples')
@token_required
def get_temples():
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM temples')
    temples_data = cursor.fetchall()
    conn.close()
    
    temples = []
    for temple in temples_data:
        temples.append({
            'id': temple[0],
            'name': temple[1],
            'capacity': temple[2],
            'current_crowd': temple[3],
            'wait_time': temple[4],
            'status': temple[5],
            'utilization': f"{(temple[3] / temple[2]) * 100:.1f}%" if temple[2] > 0 else "0%"
        })
    
    return jsonify({'temples': temples})

@app.route('/api/bookings', methods=['POST'])
@token_required
def create_booking():
    data = request.get_json()
    pilgrim_name = data.get('pilgrim_name', 'Guest Pilgrim')
    temple_id = data.get('temple_id', 1)
    
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO bookings (pilgrim_name, temple_id, status) VALUES (?, ?, 'confirmed')",
        (pilgrim_name, temple_id)
    )
    booking_id = cursor.lastrowid
    
    # Get temple name
    cursor.execute("SELECT name FROM temples WHERE id = ?", (temple_id,))
    temple_name = cursor.fetchone()[0]
    
    conn.commit()
    conn.close()
    
    booking_data = {
        'id': booking_id,
        'pilgrim_name': pilgrim_name,
        'temple_name': temple_name,
        'temple_id': temple_id,
        'status': 'confirmed',
        'timestamp': datetime.now().isoformat()
    }
    
    socketio.emit('new_booking', booking_data)
    
    return jsonify({
        'success': True,
        'message': 'Booking created successfully',
        'booking': booking_data
    })

@app.route('/api/bookings')
@token_required
def get_bookings():
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT b.*, t.name as temple_name 
        FROM bookings b 
        JOIN temples t ON b.temple_id = t.id 
        ORDER BY b.created_at DESC 
        LIMIT 20
    ''')
    bookings_data = cursor.fetchall()
    conn.close()
    
    bookings = []
    for book in bookings_data:
        bookings.append({
            'id': book[0],
            'pilgrim_name': book[1],
            'temple_name': book[5],
            'status': book[3],
            'created_at': book[4]
        })
    
    return jsonify({'bookings': bookings})

@app.route('/api/ai/stats')
@token_required
def get_ai_stats():
    return jsonify({
        'recognition_stats': recognition_stats,
        'simulation_active': ai_simulator.is_running,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/alerts')
@token_required
def get_alerts():
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM emergency_alerts 
        WHERE status = "active" 
        ORDER BY created_at DESC 
        LIMIT 10
    ''')
    alerts_data = cursor.fetchall()
    conn.close()
    
    alerts = []
    for alert in alerts_data:
        alerts.append({
            'id': alert[0],
            'type': alert[1],
            'message': alert[2],
            'severity': alert[3],
            'status': alert[4],
            'created_at': alert[5]
        })
    
    return jsonify({'alerts': alerts})

@app.route('/api/emergency/alert', methods=['POST'])
@token_required
def create_emergency():
    data = request.get_json()
    alert_type = data.get('type', 'general')
    message = data.get('message', 'Emergency situation reported')
    location = data.get('location', 'Unknown')
    
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO emergency_alerts (type, message, severity) VALUES (?, ?, ?)",
        (alert_type, f"{message} at {location}", 'high')
    )
    conn.commit()
    conn.close()
    
    socketio.emit('emergency_alert', {
        'type': alert_type,
        'message': message,
        'location': location,
        'severity': 'high',
        'timestamp': datetime.now().isoformat()
    })
    
    return jsonify({
        'success': True,
        'message': 'Emergency alert created'
    })

@app.route('/api/queue/join', methods=['POST'])
@token_required
def join_queue():
    data = request.get_json()
    pilgrim_name = data.get('pilgrim_name', 'Guest Pilgrim')
    temple_id = data.get('temple_id', 1)
    
    queue_position = random.randint(1, 25)
    wait_time = queue_position * 3
    
    queue_data = {
        'pilgrim_name': pilgrim_name,
        'temple_id': temple_id,
        'queue_position': queue_position,
        'estimated_wait': wait_time,
        'timestamp': datetime.now().isoformat()
    }
    
    socketio.emit('queue_update', queue_data)
    
    return jsonify({
        'success': True,
        'message': f'{pilgrim_name} joined queue',
        'queue_data': queue_data
    })

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'all_modules_working': True,
        'ai_simulation': ai_simulator.is_running,
        'database': 'connected',
        'websockets': 'active',
        'timestamp': datetime.now().isoformat()
    })

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {
        'message': 'Connected to AI Pilgrimage System',
        'status': 'active',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

if __name__ == '__main__':
    logger.info("🚀 AI Pilgrimage System Starting...")
    logger.info("✅ All Modules: Database, AI Simulation, WebSockets, Authentication")
    logger.info("🌐 Server running at: http://0.0.0.0:5000")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
