"""
Pilgrimage Crowd Management System - Advanced AI Detection Backend
"""

import os
import cv2
import numpy as np
import face_recognition
import sqlite3
import json
import threading
import time
import random
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import bcrypt
import jwt
from functools import wraps
import logging
import sys
from collections import defaultdict, deque
import torch
import torchvision
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pilgrimage_system.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pilgrimage_management_secret_key_2024'
app.config['DEBUG'] = True

# Enable CORS for Flutter app
CORS(app, origins=["*"])

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# JWT Configuration
JWT_SECRET = 'pilgrimage_jwt_secret_2024'
JWT_ALGORITHM = 'HS256'

# Global variables
camera = None
camera_active = False
frame_count = 0

# Database initialization
def init_db():
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Pilgrims table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pilgrims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            city TEXT,
            special_needs BOOLEAN DEFAULT FALSE,
            preferred_language TEXT,
            visit_count INTEGER DEFAULT 0,
            last_seen TIMESTAMP,
            face_encoding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            status TEXT DEFAULT 'open',
            opening_time TIME DEFAULT '06:00',
            closing_time TIME DEFAULT '21:00'
        )
    ''')
    
    # Bookings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pilgrim_id INTEGER,
            temple_id INTEGER,
            booking_time TIMESTAMP NOT NULL,
            darshan_time TIMESTAMP NOT NULL,
            status TEXT DEFAULT 'confirmed',
            qr_code TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pilgrim_id) REFERENCES pilgrims (id),
            FOREIGN KEY (temple_id) REFERENCES temples (id)
        )
    ''')
    
    # Emergency alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emergency_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            location TEXT NOT NULL,
            message TEXT NOT NULL,
            severity INTEGER DEFAULT 1,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Queue table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pilgrim_id INTEGER,
            temple_id INTEGER,
            queue_number INTEGER,
            status TEXT DEFAULT 'waiting',
            estimated_time TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pilgrim_id) REFERENCES pilgrims (id),
            FOREIGN KEY (temple_id) REFERENCES temples (id)
        )
    ''')
    
    # Detection history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temple_id INTEGER,
            people_count INTEGER,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert default admin user
    admin_password = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, password_hash, role) 
        VALUES (?, ?, ?)
    ''', ('admin', admin_password, 'admin'))
    
    # Insert default user
    user_password = bcrypt.hashpw('user123'.encode('utf-8'), bcrypt.gensalt())
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, password_hash, role) 
        VALUES (?, ?, ?)
    ''', ('user', user_password, 'user'))
    
    # Insert default temples
    temples = [
        ('Main Temple', 200, '06:00', '21:00'),
        ('North Temple', 150, '07:00', '20:00'),
        ('South Temple', 100, '08:00', '19:00')
    ]
    
    for temple in temples:
        cursor.execute('''
            INSERT OR IGNORE INTO temples (name, capacity, opening_time, closing_time) 
            VALUES (?, ?, ?, ?)
        ''', temple)
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Sample pilgrim data
sample_pilgrims = [
    {
        'id': 1,
        'name': 'Rajesh Kumar',
        'age': 45,
        'city': 'Mumbai',
        'special_needs': False,
        'preferred_language': 'Hindi',
        'visit_count': 3
    },
    {
        'id': 2,
        'name': 'Priya Sharma',
        'age': 32,
        'city': 'Delhi',
        'special_needs': False,
        'preferred_language': 'English',
        'visit_count': 2
    },
    {
        'id': 3,
        'name': 'Amit Patel',
        'age': 68,
        'city': 'Ahmedabad',
        'special_needs': True,
        'preferred_language': 'Gujarati',
        'visit_count': 5
    }
]

# Initialize data stores
temples_data = []
alerts_data = []
pilgrims_data = []
emergencies_data = []
queue_data = defaultdict(deque)
booking_data = []

recognition_stats = {
    'total_recognized': 0,
    'current_pilgrims': 0,
    'unknown_visitors': 0,
    'faces_detected': 0,
    'recognition_confidence': 0.0,
    'timestamp': datetime.now().isoformat()
}

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
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            current_user = data['username']
        except Exception as e:
            return jsonify({'error': 'Token is invalid'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            if data['role'] != 'admin':
                return jsonify({'error': 'Admin access required'}), 403
                
        except Exception as e:
            return jsonify({'error': 'Token is invalid'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

class AdvancedAIPilgrimDetector:
    """Advanced AI Pilgrim Detector with multiple models"""
    
    def __init__(self):
        # Detection settings
        self.processing_width = 640
        self.frame_skip = 3  # Process every 3rd frame for performance
        self.detection_interval = 1.5  # Seconds between detections
        
        # AI Model initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.face_detector = self.load_face_detection_model()
        self.yolo_model = self.load_yolo_model()
        
        # Detection thresholds
        self.face_confidence_threshold = 0.7
        self.person_confidence_threshold = 0.5
        
        # Tracking for smooth counts
        self.tracked_objects = []
        self.track_id = 0
        self.max_tracking_distance = 50
        self.detection_history = []
        self.history_size = 10
        
        self.cap = None
        self.is_running = False
        self.last_detection_time = 0
        self.frame_count = 0
        
        # Performance optimization
        self.last_frame = None
        self.last_detections = []

    def load_face_detection_model(self):
        """Load face detection model"""
        try:
            # Use OpenCV's DNN face detector (lightweight and accurate)
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            
            # Download model if not exists
            if not os.path.exists(model_file):
                logger.info("Downloading face detection model...")
                # Use built-in Haar cascade as fallback
                return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
            logger.info("✅ DNN Face detector loaded")
            return net
        except Exception as e:
            logger.warning(f"DNN face detector failed: {e}")
            # Fallback to Haar cascade
            return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_yolo_model(self):
        """Load YOLOv5 for person detection"""
        try:
            # Try to import ultralytics YOLO
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')  # Nano version for speed
            logger.info("✅ YOLOv8 model loaded")
            return model
        except Exception as e:
            logger.warning(f"YOLOv8 loading failed: {e}")
            try:
                # Fallback to OpenCV DNN with YOLO
                weights = "yolov3-tiny.weights"
                config = "yolov3-tiny.cfg"
                
                if os.path.exists(weights) and os.path.exists(config):
                    net = cv2.dnn.readNetFromDarknet(config, weights)
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    logger.info("✅ YOLOv3-tiny model loaded")
                    return net
                else:
                    logger.warning("YOLO weights not found, using Haar cascade for person detection")
                    return None
            except Exception as e2:
                logger.warning(f"YOLO fallback failed: {e2}")
                return None

    def initialize_camera(self):
        """Initialize laptop camera with optimal settings"""
        try:
            self.cap = cv2.VideoCapture(0)  # Laptop camera
            
            if self.cap.isOpened():
                # Set optimal camera settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 20)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                
                # Test camera
                ret, test_frame = self.cap.read()
                if ret:
                    logger.info(f"✅ Laptop camera connected - {test_frame.shape[1]}x{test_frame.shape[0]}")
                    return True
                else:
                    logger.error("❌ Camera test frame failed")
                    return False
            return False
            
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            return False

    def detect_faces(self, frame):
        """Detect faces using the available model"""
        faces = []
        try:
            if isinstance(self.face_detector, cv2.dnn_Net):
                # DNN face detection
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > self.face_confidence_threshold:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        faces.append((x1, y1, x2-x1, y2-y1))
            else:
                # Haar cascade face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
        except Exception as e:
            logger.error(f"Face detection error: {e}")
        
        return faces

    def detect_persons_yolo(self, frame):
        """Detect persons using YOLO"""
        persons = []
        try:
            if hasattr(self.yolo_model, 'predict'):
                # Ultralytics YOLOv8
                results = self.yolo_model(frame, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            if box.cls == 0 and box.conf > self.person_confidence_threshold:  # Person class
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                persons.append((x1, y1, x2-x1, y2-y1))
            elif isinstance(self.yolo_model, cv2.dnn_Net):
                # OpenCV DNN YOLO
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                self.yolo_model.setInput(blob)
                outputs = self.yolo_model.forward()
                
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        if class_id == 0 and confidence > self.person_confidence_threshold:  # Person class
                            center_x = int(detection[0] * w)
                            center_y = int(detection[1] * h)
                            width = int(detection[2] * w)
                            height = int(detection[3] * h)
                            
                            x1 = int(center_x - width/2)
                            y1 = int(center_y - height/2)
                            persons.append((x1, y1, width, height))
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
        
        return persons

    def detect_persons_haar(self, frame):
        """Fallback person detection using Haar cascade"""
        persons = []
        try:
            # Load person cascade if available
            cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            if os.path.exists(cascade_path):
                person_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bodies = person_cascade.detectMultiScale(gray, 1.1, 3)
                persons = list(bodies)
        except Exception as e:
            logger.warning(f"Haar person detection failed: {e}")
        
        return persons

    def advanced_ai_detection(self, frame):
        """Main AI detection function combining multiple models"""
        try:
            all_detections = []
            
            # 1. YOLO Person Detection (Primary)
            yolo_detections = self.detect_persons_yolo(frame)
            all_detections.extend([{'bbox': bbox, 'type': 'person', 'model': 'yolo'} for bbox in yolo_detections])
            
            # 2. Face Detection (Secondary - for close-range verification)
            if len(yolo_detections) < 10:  # Performance optimization
                face_detections = self.detect_faces(frame)
                all_detections.extend([{'bbox': bbox, 'type': 'face', 'model': 'face'} for bbox in face_detections])
            
            # 3. Haar Cascade Fallback
            if not all_detections:
                haar_detections = self.detect_persons_haar(frame)
                all_detections.extend([{'bbox': bbox, 'type': 'person', 'model': 'haar'} for bbox in haar_detections])
            
            # Apply non-maximum suppression
            filtered_detections = self.non_max_suppression(all_detections)
            
            # Calculate confidence scores
            final_detections = []
            for detection in filtered_detections:
                bbox = detection['bbox']
                x, y, w, h = bbox
                
                # Calculate confidence based on detection type
                if detection['type'] == 'face':
                    confidence = 0.9  # Faces are highly reliable
                    detection_type = "Face"
                else:
                    # Person detection confidence based on bounding box quality
                    aspect_ratio = h / w if w > 0 else 0
                    size_ratio = (w * h) / (frame.shape[0] * frame.shape[1])
                    
                    if 1.5 < aspect_ratio < 3.5 and size_ratio > 0.001:
                        confidence = 0.8
                    else:
                        confidence = 0.6
                    detection_type = "Person"
                
                final_detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'type': detection_type,
                    'model': detection['model']
                })
            
            # Update tracking for smooth counts
            self.update_tracking(final_detections)
            
            # Get smoothed count
            people_count = self.smooth_detection_count(len(final_detections))
            
            return people_count, final_detections
            
        except Exception as e:
            logger.error(f"AI detection error: {e}")
            return 0, []

    def non_max_suppression(self, detections, overlap_thresh=0.5):
        """Apply non-maximum suppression to remove overlapping detections"""
        if len(detections) == 0:
            return []
        
        boxes = np.array([det['bbox'] for det in detections])
        
        # Convert to x1, y1, x2, y2
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)  # Sort by bottom y coordinate
        
        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        
        return [detections[i] for i in pick]

    def update_tracking(self, detections):
        """Object tracking for consistent counts"""
        current_time = time.time()
        
        # Remove old tracks (older than 3 seconds)
        self.tracked_objects = [obj for obj in self.tracked_objects 
                               if current_time - obj['last_seen'] < 3.0]
        
        # Update existing tracks or create new ones
        for detection in detections:
            bbox = detection['bbox']
            center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
            
            # Find closest existing track
            min_distance = float('inf')
            best_track = None
            
            for track in self.tracked_objects:
                track_center = track['center']
                distance = np.sqrt((center[0] - track_center[0])**2 + 
                                 (center[1] - track_center[1])**2)
                
                if distance < min_distance and distance < self.max_tracking_distance:
                    min_distance = distance
                    best_track = track
            
            if best_track:
                # Update existing track
                best_track['center'] = center
                best_track['last_seen'] = current_time
                best_track['bbox'] = bbox
            else:
                # Create new track
                self.tracked_objects.append({
                    'id': self.track_id,
                    'center': center,
                    'bbox': bbox,
                    'first_seen': current_time,
                    'last_seen': current_time
                })
                self.track_id += 1

    def smooth_detection_count(self, current_count):
        """Smooth detection count using historical data"""
        self.detection_history.append(current_count)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # Use weighted average with more weight to recent detections
        if len(self.detection_history) > 0:
            weights = np.linspace(0.5, 1.5, len(self.detection_history))
            weighted_avg = np.average(self.detection_history, weights=weights)
            return int(round(weighted_avg))
        
        return current_count

    def draw_detection_overlay(self, frame, detections):
        """Draw detection overlay on frame"""
        try:
            overlay = frame.copy()
            
            for detection in detections:
                x, y, w, h = detection['bbox']
                confidence = detection['confidence']
                det_type = detection['type']
                
                # Color coding
                if det_type == "Face":
                    color = (0, 255, 0)  # Green for faces
                    label = f"Face ({confidence*100:.0f}%)"
                else:
                    color = (0, 255, 255)  # Yellow for persons
                    label = f"Person ({confidence*100:.0f}%)"
                
                # Draw bounding box
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                cv2.rectangle(overlay, (x, y-25), (x + 150, y), color, -1)
                cv2.putText(overlay, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add statistics overlay
            stats_text = f"AI Detection: {len(detections)} | Tracked: {len(self.tracked_objects)}"
            cv2.putText(overlay, stats_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(overlay, timestamp, (10, overlay.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(overlay, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return overlay
            
        except Exception as e:
            logger.error(f"Overlay drawing error: {e}")
            return frame

    def run_ai_detection(self):
        """Main AI detection loop optimized for performance"""
        if not self.initialize_camera():
            logger.error("❌ Failed to initialize camera")
            return
        
        self.is_running = True
        logger.info("🚀 Starting Advanced AI Pilgrim Detection")
        logger.info("⚡ Models: YOLO + Face Detection + Tracking")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame")
                    time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                
                # Skip frames for performance (process every 3rd frame)
                if self.frame_count % self.frame_skip != 0:
                    # Show frame without processing
                    cv2.imshow('AI Pilgrim Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Throttle detection frequency
                current_time = time.time()
                if current_time - self.last_detection_time < self.detection_interval:
                    cv2.imshow('AI Pilgrim Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                self.last_detection_time = current_time
                
                # Perform AI detection
                people_count, detections = self.advanced_ai_detection(frame)
                
                # Send data to frontend via SocketIO
                self.send_detection_data(people_count, detections)
                
                # Draw visualizations
                overlay_frame = self.draw_detection_overlay(frame, detections)
                
                # Display frame
                cv2.imshow('AI Pilgrim Detection', overlay_frame)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            logger.info("🛑 Detection stopped by user")
        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")
        finally:
            self.cleanup()

    def send_detection_data(self, people_count, detections):
        """Send detection data to frontend"""
        try:
            avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0.5
            
            # Update recognition stats
            global recognition_stats
            recognition_stats.update({
                'faces_detected': len([d for d in detections if d['type'] == 'Face']),
                'current_pilgrims': people_count,
                'unknown_visitors': 0,  # AI detection doesn't distinguish known/unknown
                'recognition_confidence': float(avg_confidence),
                'timestamp': datetime.now().isoformat()
            })
            
            # Send via SocketIO
            socketio.emit('recognition_stats', recognition_stats)
            
            # Update temple data
            self.update_temple_data(people_count, avg_confidence)
            
            logger.info(f"🤖 AI DETECTION: {people_count} pilgrims (Confidence: {avg_confidence:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error sending detection data: {e}")

    def update_temple_data(self, people_count, confidence):
        """Update temple data with AI detection results"""
        try:
            conn = sqlite3.connect('pilgrimage.db')
            cursor = conn.cursor()
            
            # Update main temple
            cursor.execute("SELECT id, capacity FROM temples WHERE name = 'Main Temple'")
            temple = cursor.fetchone()
            
            if temple:
                temple_id, capacity = temple
                
                # Calculate wait time (2 minutes per person)
                wait_time = people_count * 2
                
                # Update temple
                cursor.execute(
                    "UPDATE temples SET current_crowd = ?, wait_time = ? WHERE id = ?",
                    (people_count, wait_time, temple_id)
                )
                
                # Save to detection history
                cursor.execute(
                    "INSERT INTO detection_history (temple_id, people_count, confidence) VALUES (?, ?, ?)",
                    (temple_id, people_count, confidence)
                )
                
                conn.commit()
                
                # Send temple update
                temple_data = {
                    'id': temple_id,
                    'name': 'Main Temple',
                    'current_crowd': people_count,
                    'capacity': capacity,
                    'wait_time': wait_time,
                    'status': 'open',
                    'utilization': (people_count / capacity) * 100,
                    'density_level': self.get_density_level(people_count, capacity),
                    'timestamp': datetime.now().isoformat()
                }
                
                socketio.emit('temple_update', temple_data)
                
                # Check for alerts
                self.check_alerts(people_count, capacity)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error updating temple data: {e}")

    def get_density_level(self, crowd_count, capacity):
        """Get density level based on crowd count"""
        utilization = (crowd_count / capacity) * 100
        if utilization >= 85:
            return 'CRITICAL'
        elif utilization >= 70:
            return 'HIGH'
        elif utilization >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'

    def check_alerts(self, crowd_count, capacity):
        """Check and send alerts based on crowd density"""
        utilization = (crowd_count / capacity) * 100
        
        if utilization > 90:
            alert = {
                'id': len(alerts_data) + 1,
                'type': 'overcrowding',
                'location': 'Main Temple',
                'message': f'🚨 CRITICAL: Temple at {utilization:.1f}% capacity! {crowd_count} people detected.',
                'severity': 5,
                'timestamp': datetime.now().isoformat()
            }
            alerts_data.append(alert)
            socketio.emit('alert_update', alert)
            
            # Emergency alert
            emergency = {
                'id': len(emergencies_data) + 1,
                'type': 'overcrowding',
                'location': 'Main Temple',
                'message': f'EMERGENCY: Critical overcrowding! {crowd_count} people in temple.',
                'severity': 5,
                'timestamp': datetime.now().isoformat(),
                'status': 'active'
            }
            emergencies_data.append(emergency)
            socketio.emit('emergency_alert', emergency)
            
            # Save to database
            conn = sqlite3.connect('pilgrimage.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO emergency_alerts (type, location, message, severity, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                emergency['type'],
                emergency['location'],
                emergency['message'],
                emergency['severity'],
                emergency['status']
            ))
            conn.commit()
            conn.close()

    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("🛑 AI detection system shut down")

# Initialize AI detector
ai_detector = AdvancedAIPilgrimDetector()

# Initialize managers
class QueueManager:
    """Manage temple queues"""
    
    def __init__(self):
        self.queues = defaultdict(deque)
        self.queue_counters = defaultdict(int)
        self.processing_rate = 2  # persons per minute
    
    def add_to_queue(self, pilgrim_id, temple_id):
        """Add pilgrim to queue"""
        queue_number = self.queue_counters[temple_id] + 1
        self.queue_counters[temple_id] = queue_number
        
        queue_entry = {
            'pilgrim_id': pilgrim_id,
            'temple_id': temple_id,
            'queue_number': queue_number,
            'status': 'waiting',
            'estimated_time': datetime.now() + timedelta(minutes=len(self.queues[temple_id]) * self.processing_rate),
            'created_at': datetime.now().isoformat()
        }
        
        self.queues[temple_id].append(queue_entry)
        return queue_entry
    
    def process_next(self, temple_id):
        """Process next pilgrim in queue"""
        if self.queues[temple_id]:
            pilgrim = self.queues[temple_id].popleft()
            pilgrim['status'] = 'processed'
            pilgrim['processed_at'] = datetime.now().isoformat()
            return pilgrim
        return None
    
    def get_queue_status(self, temple_id):
        """Get current queue status"""
        queue_list = list(self.queues[temple_id])
        return {
            'temple_id': temple_id,
            'current_queue': len(queue_list),
            'wait_time': len(queue_list) * self.processing_rate,
            'queue_list': queue_list[:10]  # Return first 10 in queue
        }

class CrowdPredictor:
    """AI crowd prediction system"""
    
    def __init__(self):
        self.historical_data = defaultdict(list)
    
    def add_data_point(self, temple_id, crowd_count):
        """Add crowd data point"""
        timestamp = datetime.now()
        self.historical_data[temple_id].append({
            'timestamp': timestamp,
            'crowd_count': crowd_count
        })
        
        # Keep only last 24 hours data
        cutoff_time = timestamp - timedelta(hours=24)
        self.historical_data[temple_id] = [
            point for point in self.historical_data[temple_id] 
            if point['timestamp'] > cutoff_time
        ]
    
    def predict_crowd(self, temple_id):
        """Predict future crowd levels"""
        data = self.historical_data[temple_id]
        if len(data) < 10:
            return {'prediction': 'Insufficient data', 'confidence': 0.0}
        
        recent_data = data[-30:]  # Last 30 data points
        crowds = [point['crowd_count'] for point in recent_data]
        
        if len(crowds) < 2:
            return {'prediction': 'Stable', 'confidence': 0.5}
        
        # Continuing from the previous code...

        current_crowd = crowds[-1]
        avg_crowd = np.mean(crowds)
        
        if current_crowd > avg_crowd * 1.3:
            prediction = 'Increasing'
            confidence = min(0.9, (current_crowd - avg_crowd) / avg_crowd)
        elif current_crowd < avg_crowd * 0.7:
            prediction = 'Decreasing'
            confidence = min(0.9, (avg_crowd - current_crowd) / avg_crowd)
        else:
            prediction = 'Stable'
            confidence = 0.6
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'estimated_crowd_1h': int(current_crowd * (1.1 if prediction == 'Increasing' else 0.9)),
            'recommendation': self.get_recommendation(prediction, current_crowd)
        }
    
    def get_recommendation(self, prediction, current_crowd):
        """Get recommendation based on prediction"""
        if prediction == 'Increasing' and current_crowd > 150:
            return "Consider opening additional entry points"
        elif prediction == 'Decreasing' and current_crowd < 50:
            return "Normal crowd levels expected"
        else:
            return "Maintain current operations"

# Initialize managers
queue_manager = QueueManager()
crowd_predictor = CrowdPredictor()

def camera_processing_thread():
    """Main camera processing thread using AI detection"""
    global camera_active
    
    logger.info("🚀 Starting AI-powered camera processing")
    ai_detector.run_ai_detection()

# Authentication endpoints
@app.route('/api/login', methods=['POST'])
def login():
    """User login"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and bcrypt.checkpw(password.encode('utf-8'), user[2]):
        token = jwt.encode({
            'username': username,
            'role': user[3],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, JWT_SECRET, algorithm=JWT_ALGORITHM)
        
        return jsonify({
            'token': token,
            'user': {
                'username': username,
                'role': user[3]
            }
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

# Temple endpoints
@app.route('/api/temples', methods=['GET'])
@token_required
def get_temples(current_user):
    """Get all temples"""
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM temples')
    temples = cursor.fetchall()
    conn.close()
    
    temple_list = []
    for temple in temples:
        temple_id, name, capacity, current_crowd, wait_time, status, opening_time, closing_time = temple
        
        # Get crowd prediction
        prediction = crowd_predictor.predict_crowd(temple_id)
        
        temple_list.append({
            'id': temple_id,
            'name': name,
            'capacity': capacity,
            'current_crowd': current_crowd,
            'wait_time': wait_time,
            'status': status,
            'opening_time': opening_time,
            'closing_time': closing_time,
            'utilization': (current_crowd / capacity) * 100 if capacity > 0 else 0,
            'density_level': get_density_level(current_crowd, capacity),
            'prediction': prediction
        })
    
    return jsonify({'temples': temple_list})

def get_density_level(crowd_count, capacity):
    """Get density level based on crowd count"""
    if capacity == 0:
        return 'LOW'
    
    utilization = (crowd_count / capacity) * 100
    if utilization >= 85:
        return 'CRITICAL'
    elif utilization >= 70:
        return 'HIGH'
    elif utilization >= 40:
        return 'MEDIUM'
    else:
        return 'LOW'

# Booking endpoints
@app.route('/api/bookings', methods=['POST'])
@token_required
def create_booking(current_user):
    """Create virtual darshan booking"""
    data = request.get_json()
    temple_id = data.get('temple_id')
    pilgrim_name = data.get('pilgrim_name')
    darshan_time = data.get('darshan_time')
    
    if not temple_id or not pilgrim_name or not darshan_time:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        darshan_datetime = datetime.fromisoformat(darshan_time.replace('Z', '+00:00'))
        if darshan_datetime < datetime.now():
            return jsonify({'error': 'Darshan time cannot be in the past'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    
    # Generate QR code
    qr_code = f"T{temple_id}_{pilgrim_name}_{int(datetime.now().timestamp())}"
    
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    
    # Create a temporary pilgrim entry
    cursor.execute('''
        INSERT OR IGNORE INTO pilgrims (name, visit_count, last_seen) 
        VALUES (?, 1, ?)
    ''', (pilgrim_name, datetime.now().isoformat()))
    
    pilgrim_id = cursor.lastrowid
    
    cursor.execute('''
        INSERT INTO bookings (pilgrim_id, temple_id, booking_time, darshan_time, qr_code)
        VALUES (?, ?, ?, ?, ?)
    ''', (pilgrim_id, temple_id, datetime.now().isoformat(), darshan_datetime.isoformat(), qr_code))
    
    booking_id = cursor.lastrowid
    conn.commit()
    
    # Get temple name
    cursor.execute('SELECT name FROM temples WHERE id = ?', (temple_id,))
    temple_name = cursor.fetchone()[0]
    conn.close()
    
    booking_data = {
        'id': booking_id,
        'pilgrim_name': pilgrim_name,
        'temple_id': temple_id,
        'temple_name': temple_name,
        'darshan_time': darshan_datetime.isoformat(),
        'qr_code': qr_code,
        'status': 'confirmed'
    }
    
    return jsonify({'booking': booking_data})

@app.route('/api/bookings', methods=['GET'])
@token_required
def get_bookings(current_user):
    """Get user bookings"""
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT b.*, t.name as temple_name, p.name as pilgrim_name
        FROM bookings b 
        JOIN temples t ON b.temple_id = t.id 
        JOIN pilgrims p ON b.pilgrim_id = p.id
        ORDER BY b.darshan_time DESC
        LIMIT 20
    ''')
    bookings = cursor.fetchall()
    conn.close()
    
    booking_list = []
    for booking in bookings:
        booking_list.append({
            'id': booking[0],
            'pilgrim_name': booking[8],
            'temple_name': booking[7],
            'darshan_time': booking[4],
            'qr_code': booking[6],
            'status': booking[5]
        })
    
    return jsonify({'bookings': booking_list})

# Queue endpoints
@app.route('/api/queue/<int:temple_id>/join', methods=['POST'])
@token_required
def join_queue(current_user, temple_id):
    """Join temple queue"""
    data = request.get_json()
    pilgrim_name = data.get('pilgrim_name', 'Guest Pilgrim')
    
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    
    # Create or get pilgrim
    cursor.execute('''
        INSERT OR IGNORE INTO pilgrims (name, visit_count, last_seen) 
        VALUES (?, 1, ?)
    ''', (pilgrim_name, datetime.now().isoformat()))
    
    pilgrim_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    queue_entry = queue_manager.add_to_queue(pilgrim_id, temple_id)
    
    socketio.emit('queue_update', {
        'temple_id': temple_id,
        'action': 'joined',
        'queue_entry': queue_entry,
        'timestamp': datetime.now().isoformat()
    })
    
    return jsonify({'queue_entry': queue_entry})

@app.route('/api/queue/<int:temple_id>', methods=['GET'])
@token_required
def get_queue_status(current_user, temple_id):
    """Get queue status"""
    queue_status = queue_manager.get_queue_status(temple_id)
    
    # Add temple info
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, capacity FROM temples WHERE id = ?', (temple_id,))
    temple_info = cursor.fetchone()
    conn.close()
    
    if temple_info:
        queue_status['temple_name'] = temple_info[0]
        queue_status['capacity'] = temple_info[1]
    
    return jsonify(queue_status)

@app.route('/api/queue/<int:temple_id>/next', methods=['POST'])
@admin_required
def process_next_in_queue(temple_id):
    """Process next pilgrim in queue (Admin only)"""
    processed_pilgrim = queue_manager.process_next(temple_id)
    
    if processed_pilgrim:
        socketio.emit('queue_update', {
            'temple_id': temple_id,
            'action': 'processed',
            'pilgrim': processed_pilgrim,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({'success': True, 'processed_pilgrim': processed_pilgrim})
    else:
        return jsonify({'success': False, 'message': 'Queue is empty'})

# Emergency alerts endpoints
@app.route('/api/emergency-alerts', methods=['GET'])
@token_required
def get_emergency_alerts(current_user):
    """Get emergency alerts"""
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM emergency_alerts 
        WHERE status = "active" 
        ORDER BY created_at DESC 
        LIMIT 10
    ''')
    alerts = cursor.fetchall()
    conn.close()
    
    alert_list = []
    for alert in alerts:
        alert_list.append({
            'id': alert[0],
            'type': alert[1],
            'location': alert[2],
            'message': alert[3],
            'severity': alert[4],
            'timestamp': alert[6]
        })
    
    return jsonify({'alerts': alert_list})

@app.route('/api/emergency-alerts/<int:alert_id>/resolve', methods=['POST'])
@admin_required
def resolve_emergency_alert(alert_id):
    """Resolve emergency alert (Admin only)"""
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE emergency_alerts SET status = "resolved" WHERE id = ?
    ''', (alert_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Alert resolved'})

# Pilgrim statistics endpoints
@app.route('/api/statistics/daily', methods=['GET'])
@token_required
def get_daily_statistics(current_user):
    """Get daily pilgrimage statistics"""
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    
    # Today's date
    today = datetime.now().date()
    
    # Total pilgrims today
    cursor.execute('''
        SELECT COUNT(DISTINCT pilgrim_id) FROM bookings 
        WHERE DATE(darshan_time) = ? AND status = 'confirmed'
    ''', (today,))
    today_pilgrims = cursor.fetchone()[0]
    
    # Current visitors (from detection system)
    cursor.execute('SELECT current_crowd FROM temples WHERE id = 1')
    current_visitors = cursor.fetchone()[0] or 0
    
    # Total bookings today
    cursor.execute('SELECT COUNT(*) FROM bookings WHERE DATE(booking_time) = ?', (today,))
    today_bookings = cursor.fetchone()[0]
    
    # Queue statistics
    cursor.execute('SELECT COUNT(*) FROM queue WHERE DATE(created_at) = ? AND status = "waiting"', (today,))
    in_queue = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'date': today.isoformat(),
        'today_pilgrims': today_pilgrims,
        'current_visitors': current_visitors,
        'today_bookings': today_bookings,
        'pilgrims_in_queue': in_queue,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/statistics/history', methods=['GET'])
@admin_required
def get_detection_history(current_user):
    """Get detection history (Admin only)"""
    hours = request.args.get('hours', 24, type=int)
    
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    cursor.execute('''
        SELECT timestamp, people_count, confidence 
        FROM detection_history 
        WHERE timestamp > ? 
        ORDER BY timestamp DESC
    ''', (cutoff_time.isoformat(),))
    
    history = cursor.fetchall()
    conn.close()
    
    history_list = []
    for record in history:
        history_list.append({
            'timestamp': record[0],
            'people_count': record[1],
            'confidence': record[2]
        })
    
    return jsonify({'history': history_list})

# Admin endpoints
@app.route('/api/admin/start-camera', methods=['POST'])
@admin_required
def start_camera():
    """Start AI camera detection (Admin only)"""
    global camera_active
    
    if not camera_active:
        # Start AI detection in a separate thread
        detection_thread = threading.Thread(target=camera_processing_thread, daemon=True)
        detection_thread.start()
        
        # Wait for initialization
        time.sleep(3)
        
        return jsonify({
            'success': True,
            'message': 'AI Camera detection started successfully',
            'models': 'YOLO + Face Detection + Tracking',
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'success': False,
            'message': 'AI Detection is already running'
        })

@app.route('/api/admin/stop-camera', methods=['POST'])
@admin_required
def stop_camera():
    """Stop AI camera detection (Admin only)"""
    global camera_active
    ai_detector.is_running = False
    camera_active = False
    
    return jsonify({
        'success': True, 
        'message': 'AI Detection stopped successfully'
    })

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_admin_stats():
    """Get admin statistics"""
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    
    # Today's pilgrims count
    today = datetime.now().date()
    cursor.execute('''
        SELECT COUNT(DISTINCT pilgrim_id) FROM bookings WHERE DATE(darshan_time) = ?
    ''', (today,))
    today_pilgrims = cursor.fetchone()[0]
    
    # Total pilgrims
    cursor.execute('SELECT COUNT(DISTINCT id) FROM pilgrims')
    total_pilgrims = cursor.fetchone()[0]
    
    # Current visitors from all temples
    cursor.execute('SELECT SUM(current_crowd) FROM temples')
    current_visitors = cursor.fetchone()[0] or 0
    
    # Active alerts
    cursor.execute('SELECT COUNT(*) FROM emergency_alerts WHERE status = "active"')
    active_alerts = cursor.fetchone()[0]
    
    # System status
    cursor.execute('SELECT COUNT(*) FROM detection_history WHERE timestamp > ?', 
                  ((datetime.now() - timedelta(minutes=5)).isoformat(),))
    recent_detections = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'today_pilgrims': today_pilgrims,
        'total_pilgrims': total_pilgrims,
        'current_visitors': current_visitors,
        'active_alerts': active_alerts,
        'system_status': 'active' if recent_detections > 0 else 'inactive',
        'ai_detection_active': ai_detector.is_running,
        'recognition_stats': recognition_stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/admin/simulate-crowd', methods=['POST'])
@admin_required
def simulate_crowd():
    """Simulate crowd data for testing (Admin only)"""
    data = request.get_json()
    crowd_size = data.get('size', 50)
    
    # Update temple data with simulated crowd
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    
    cursor.execute("UPDATE temples SET current_crowd = ? WHERE id = 1", (crowd_size,))
    cursor.execute(
        "INSERT INTO detection_history (temple_id, people_count, confidence) VALUES (?, ?, ?)",
        (1, crowd_size, 0.85)
    )
    
    conn.commit()
    conn.close()
    
    # Send update via socket
    temple_data = {
        'id': 1,
        'name': 'Main Temple',
        'current_crowd': crowd_size,
        'capacity': 200,
        'wait_time': crowd_size * 2,
        'utilization': (crowd_size / 200) * 100,
        'density_level': get_density_level(crowd_size, 200),
        'timestamp': datetime.now().isoformat()
    }
    
    socketio.emit('temple_update', temple_data)
    
    return jsonify({
        'success': True,
        'message': f'Simulated crowd of {crowd_size} pilgrims',
        'temple_data': temple_data
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"✅ Client connected: {request.sid}")
    emit('connection_status', {
        'status': 'connected',
        'message': 'Connected to Pilgrimage AI System',
        'timestamp': datetime.now().isoformat()
    })
    
    # Send current state
    emit('recognition_stats', recognition_stats)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"🔌 Client disconnected: {request.sid}")

@socketio.on('request_temple_update')
def handle_temple_update():
    """Handle temple update request"""
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM temples')
    temples = cursor.fetchall()
    conn.close()
    
    for temple in temples:
        temple_id, name, capacity, current_crowd, wait_time, status, opening_time, closing_time = temple
        prediction = crowd_predictor.predict_crowd(temple_id)
        
        temple_data = {
            'id': temple_id,
            'name': name,
            'capacity': capacity,
            'current_crowd': current_crowd,
            'wait_time': wait_time,
            'status': status,
            'utilization': (current_crowd / capacity) * 100,
            'density_level': get_density_level(current_crowd, capacity),
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        emit('temple_update', temple_data)

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ai_detection_active': ai_detector.is_running,
        'camera_available': ai_detector.cap.isOpened() if ai_detector.cap else False,
        'database_connected': True,
        'version': '2.0.0'
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("🎯 Starting Advanced Pilgrimage AI Management System...")
    logger.info("🌐 Server will be available at: http://localhost:5000")
    logger.info("🤖 AI Features: YOLO Detection + Face Recognition + Crowd Prediction")
    logger.info("📊 System Includes: Booking, Queue Management, Emergency Alerts")
    
    # Load initial temple data
    conn = sqlite3.connect('pilgrimage.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM temples')
    temples_data = cursor.fetchall()
    conn.close()
    
    try:
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=5000, 
            debug=False, 
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
    finally:
        ai_detector.cleanup()
        logger.info("🎊 AI Pilgrimage System shutdown completed")
