"""
Pilgrimage Crowd Management System - Advanced AI Detection Backend
with Integrated High-Performance Detection
"""

import os
import cv2
import numpy as np

import sqlite3
import json
import threading
import time
import random
import requests
import base64
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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
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
camera_active = False

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
    
    # Emergency logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emergency_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            user_data TEXT,
            description TEXT,
            location TEXT,
            severity TEXT,
            status TEXT DEFAULT 'active',
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

# Initialize data stores
alerts_data = []
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

class IntegratedAIPilgrimDetector:
    """Integrated AI Pilgrim Detector with enhanced multi-model detection"""
    
    def __init__(self):
        # Detection settings optimized for smooth FPS
        self.processing_width = 480
        self.frame_skip = 4
        self.detection_interval = 2.0
        
        # AI Model initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained models
        self.face_detector = self.load_face_detection_model()
        self.person_detector = self.load_person_detection_model()
        self.yolo_model = self.load_yolo_model()
        
        # Detection thresholds
        self.face_confidence_threshold = 0.7
        self.person_confidence_threshold = 0.6
        self.yolo_confidence_threshold = 0.5
        
        # Tracking
        self.tracked_objects = []
        self.track_id = 0
        self.max_tracking_distance = 50
        self.detection_history = []
        self.history_size = 15
        
        self.cap = None
        self.is_running = False
        self.last_detection_time = 0
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.frames_processed = 0
        
        # Performance monitoring
        self.performance_stats = {
            'fps': 0,
            'detection_time': 0,
            'faces_detected': 0,
            'persons_detected': 0,
            'total_detections': 0
        }

    def load_face_detection_model(self):
        """Load MTCNN or RetinaFace for face detection"""
        try:
            # Try to load RetinaFace (more accurate)
            from retinaface import RetinaFace
            logger.info("✅ RetinaFace model loaded")
            return 'retinaface'
        except:
            try:
                from facenet_pytorch import MTCNN
                mtcnn = MTCNN(keep_all=True, device=self.device, min_face_size=20)
                logger.info("✅ MTCNN model loaded")
                return mtcnn
            except:
                # Fallback to OpenCV DNN face detector
                model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
                config_file = "deploy.prototxt"
                try:
                    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
                    logger.info("✅ OpenCV DNN Face detector loaded")
                    return net
                except:
                    logger.warning("❌ Could not load advanced face detector, using Haar cascades")
                    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_person_detection_model(self):
        """Load Faster R-CNN or YOLO for person detection"""
        try:
            # Load Faster R-CNN pre-trained on COCO
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval().to(self.device)
            logger.info("✅ Faster R-CNN model loaded")
            return model
        except Exception as e:
            logger.warning(f"Faster R-CNN loading failed: {e}")
            return None

    def load_yolo_model(self):
        """Load YOLOv5 for person detection"""
        try:
            # Load YOLOv5 (much faster and accurate)
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.conf = self.yolo_confidence_threshold  # Confidence threshold
            model.classes = [0]  # Only detect persons (class 0 in COCO)
            logger.info("✅ YOLOv5 model loaded")
            return model
        except Exception as e:
            logger.warning(f"YOLOv5 loading failed: {e}")
            return None

    def initialize_camera_optimized(self):
        """Initialize camera with optimal settings"""
        try:
            # Try multiple camera sources
            camera_sources = [
                
                "http://192.0.0.4:8080/video",
                "http://192.168.1.100:8080/video"  # Add more if needed
            ]
            
            for source in camera_sources:
                logger.info(f"🔍 Trying camera source: {source}")
                self.cap = cv2.VideoCapture(source)
                
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                    self.cap.set(cv2.CAP_PROP_FPS, 20)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test camera with timeout
                    test_start = time.time()
                    while time.time() - test_start < 5:
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"✅ Camera connected at source {source}")
                            return True
                        time.sleep(0.1)
                    
                    self.cap.release()
            
            logger.error("❌ No working camera source available")
            return False
            
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            return False

    def detect_faces_dnn(self, frame):
        """High-accuracy face detection using DNN"""
        try:
            if isinstance(self.face_detector, cv2.dnn_Net):
                # OpenCV DNN face detection
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()
                
                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > self.face_confidence_threshold:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        
                        if x2 > x1 and y2 > y1:  # Valid bounding box
                            faces.append((x1, y1, x2-x1, y2-y1))
                
                logger.debug(f"DNN detected {len(faces)} faces")
                return faces
            else:
                return []
        except Exception as e:
            logger.error(f"DNN face detection error: {e}")
            return []

    def detect_faces_mtcnn(self, frame):
        """Face detection using MTCNN"""
        try:
            if hasattr(self.face_detector, 'detect'):
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Detect faces
                boxes, probs = self.face_detector.detect(pil_image)
                
                faces = []
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        if probs[i] > self.face_confidence_threshold:
                            x1, y1, x2, y2 = box.astype(int)
                            faces.append((x1, y1, x2-x1, y2-y1))
                
                logger.debug(f"MTCNN detected {len(faces)} faces")
                return faces
            else:
                return []
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return []

    def detect_persons_yolo(self, frame):
        """Person detection using YOLOv5"""
        try:
            if self.yolo_model is not None:
                # YOLOv5 detection
                results = self.yolo_model(frame)
                detections = results.pandas().xyxy[0]
                
                persons = []
                for _, detection in detections.iterrows():
                    if detection['confidence'] > self.yolo_confidence_threshold and detection['class'] == 0:
                        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), \
                                        int(detection['xmax']), int(detection['ymax'])
                        persons.append((x1, y1, x2-x1, y2-y1))
                
                logger.debug(f"YOLO detected {len(persons)} persons")
                return persons
            return []
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []

    def detect_persons_faster_rcnn(self, frame):
        """Person detection using Faster R-CNN"""
        try:
            if self.person_detector is not None:
                # Convert frame to tensor
                image_tensor = torchvision.transforms.functional.to_tensor(frame).to(self.device)
                
                with torch.no_grad():
                    predictions = self.person_detector([image_tensor])
                
                persons = []
                boxes = predictions[0]['boxes'].cpu().numpy()
                labels = predictions[0]['labels'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()
                
                for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                    if label == 1 and score > self.person_confidence_threshold:  # Person class
                        x1, y1, x2, y2 = box.astype(int)
                        persons.append((x1, y1, x2-x1, y2-y1))
                
                logger.debug(f"Faster R-CNN detected {len(persons)} persons")
                return persons
            return []
        except Exception as e:
            logger.error(f"Faster R-CNN detection error: {e}")
            return []

    def advanced_ai_detection(self, frame):
        """Main AI detection function combining multiple models"""
        start_time = time.time()
        
        try:
            all_detections = []
            
            # 1. YOLOv5 Person Detection (Primary - fastest and most accurate)
            yolo_detections = self.detect_persons_yolo(frame)
            all_detections.extend([{'bbox': bbox, 'type': 'person', 'model': 'yolo'} for bbox in yolo_detections])
            
            # 2. Face Detection (Secondary - for close-range verification)
            if len(yolo_detections) < 5:  # Only run if few persons detected (performance optimization)
                face_detections = self.detect_faces_dnn(frame)
                all_detections.extend([{'bbox': bbox, 'type': 'face', 'model': 'dnn'} for bbox in face_detections])
            
            # 3. Faster R-CNN (Fallback - most accurate but slower)
            if not all_detections:  # Only run if other methods fail
                frcnn_detections = self.detect_persons_faster_rcnn(frame)
                all_detections.extend([{'bbox': bbox, 'type': 'person', 'model': 'frcnn'} for bbox in frcnn_detections])
            
            # Apply non-maximum suppression
            filtered_detections = self.non_max_suppression(all_detections)
            
            # Calculate confidence scores
            final_detections = []
            for detection in filtered_detections:
                bbox = detection['bbox']
                x, y, w, h = bbox
                
                # Calculate confidence based on detection type and size
                if detection['type'] == 'face':
                    confidence = 0.9  # Faces are highly reliable
                    detection_type = "Face"
                else:
                    # Person detection confidence based on bounding box quality
                    aspect_ratio = h / w if w > 0 else 0
                    size_ratio = (w * h) / (frame.shape[0] * frame.shape[1])
                    
                    if 1.5 < aspect_ratio < 3.5 and size_ratio > 0.001:
                        confidence = 0.85
                    else:
                        confidence = 0.7
                    detection_type = "Person"
                
                # Estimate distance
                box_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                
                if box_area > frame_area * 0.05:
                    distance = "Very Close"
                elif box_area > frame_area * 0.01:
                    distance = "Close"
                elif box_area > frame_area * 0.002:
                    distance = "Medium"
                else:
                    distance = "Far"
                
                final_detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'distance': distance,
                    'type': detection_type,
                    'model': detection['model']
                })
            
            # Update tracking
            self.update_tracking(final_detections)
            
            # Smooth count
            people_count = self.smooth_detection_count(len(final_detections))
            
            # Update performance stats
            detection_time = time.time() - start_time
            face_count = len([d for d in final_detections if d['type'] == 'Face'])
            person_count = len([d for d in final_detections if d['type'] == 'Person'])
            
            self.performance_stats.update({
                'detection_time': detection_time,
                'faces_detected': face_count,
                'persons_detected': person_count,
                'total_detections': len(final_detections)
            })
            
            logger.debug(f"Integrated AI Detection: {len(final_detections)} objects in {detection_time:.3f}s")
            return people_count, final_detections
            
        except Exception as e:
            logger.error(f"AI detection error: {e}")
            return 0, []

    def non_max_suppression(self, detections, overlap_thresh=0.5):
        """Apply non-maximum suppression to remove overlapping detections"""
        if len(detections) == 0:
            return []
        
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([0.9 if det['type'] == 'face' else 0.8 for det in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)
        
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
        
        logger.debug(f"NMS reduced {len(detections)} detections to {len(pick)}")
        return [detections[i] for i in pick]

    def update_tracking(self, detections):
        """Object tracking for consistent counts"""
        current_time = time.time()
        
        self.tracked_objects = [obj for obj in self.tracked_objects 
                               if current_time - obj['last_seen'] < 3.0]
        
        for detection in detections:
            bbox = detection['bbox']
            center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
            
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
                best_track['center'] = center
                best_track['last_seen'] = current_time
                best_track['bbox'] = bbox
            else:
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
        
        weights = np.linspace(0.5, 1.5, len(self.detection_history))
        weighted_avg = np.average(self.detection_history, weights=weights)
        
        return int(round(weighted_avg))

    def calculate_fps(self):
        """Calculate current FPS"""
        self.frames_processed += 1
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        
        if time_diff >= 1.0:
            self.fps = self.frames_processed / time_diff
            self.frames_processed = 0
            self.last_fps_time = current_time
            self.performance_stats['fps'] = self.fps

    def draw_enhanced_visualizations(self, frame, detections):
        """Draw enhanced AI detection visualizations"""
        try:
            overlay = frame.copy()
            
            for detection in detections:
                x, y, w, h = detection['bbox']
                confidence = detection['confidence']
                distance = detection['distance']
                det_type = detection['type']
                model = detection['model']
                
                if det_type == "Face":
                    color = (0, 255, 0)  # Green for faces
                else:
                    if distance == "Very Close":
                        color = (0, 255, 255)  # Yellow
                    elif distance == "Close":
                        color = (0, 165, 255)  # Orange
                    elif distance == "Medium":
                        color = (0, 100, 255)  # Light red
                    else:
                        color = (0, 0, 255)  # Red
                
                # Draw bounding box
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Draw info text
                info_text = f"{det_type} {distance} ({confidence*100:.0f}%)"
                cv2.putText(overlay, info_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(overlay, f"Model: {model}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Enhanced statistics overlay
            face_count = len([d for d in detections if d['type'] == 'Face'])
            person_count = len([d for d in detections if d['type'] == 'Person'])
            
            stats_text = f"FPS: {self.fps:.1f} | Faces: {face_count} | Persons: {person_count} | Tracked: {len(self.tracked_objects)}"
            cv2.putText(overlay, stats_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            detection_time_text = f"Detection: {self.performance_stats['detection_time']*1000:.1f}ms"
            cv2.putText(overlay, detection_time_text, (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(overlay, "INTEGRATED AI DETECTION - Press 'q' to quit", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(overlay, timestamp, (10, overlay.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return overlay
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return frame

    def run_integrated_detection(self):
        """Main integrated detection loop"""
        if not self.initialize_camera_optimized():
            logger.error("❌ Failed to initialize camera")
            return
        
        self.is_running = True
        logger.info("🚀 Starting Integrated AI Pilgrim Detection")
        logger.info("⚡ Models: YOLOv5 + DNN Faces + Faster R-CNN + Tracking")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame - retrying...")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                self.calculate_fps()
                
                # Skip frames for performance
                if self.frame_count % self.frame_skip != 0:
                    display_frame = self.draw_enhanced_visualizations(frame, [])
                    cv2.imshow('Integrated AI Pilgrim Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Throttle detection frequency
                current_time = time.time()
                if current_time - self.last_detection_time < self.detection_interval:
                    display_frame = self.draw_enhanced_visualizations(frame, [])
                    cv2.imshow('Integrated AI Pilgrim Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                self.last_detection_time = current_time
                
                # Perform integrated AI detection
                people_count, detections = self.advanced_ai_detection(frame)
                
                # Send data to frontend
                self.send_detection_data(people_count, detections)
                
                # Draw enhanced visualizations
                overlay_frame = self.draw_enhanced_visualizations(frame, detections)
                
                # Stream to frontend with quality adjustment
                encode_quality = 60 if self.fps < 15 else 40
                _, buffer = cv2.imencode('.jpg', overlay_frame, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                socketio.emit('live_frame', {
                    'image': jpg_as_text,
                    'timestamp': datetime.now().isoformat(),
                    'detections': len(detections),
                    'faces_detected': self.performance_stats['faces_detected'],
                    'persons_detected': self.performance_stats['persons_detected'],
                    'fps': self.fps
                })
                
                # Display frame
                cv2.imshow('Integrated AI Pilgrim Detection', overlay_frame)
                
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
            face_count = self.performance_stats['faces_detected']
            person_count = self.performance_stats['persons_detected']
            avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0.5
            
            # Update recognition stats
            global recognition_stats
            recognition_stats.update({
                'faces_detected': face_count,
                'current_pilgrims': people_count,
                'unknown_visitors': 0,  # AI detection doesn't distinguish known/unknown
                'recognition_confidence': float(avg_confidence),
                'timestamp': datetime.now().isoformat()
            })
            
            # Send via SocketIO
            socketio.emit('recognition_stats', recognition_stats)
            
            # Update temple data
            self.update_temple_data(people_count, avg_confidence)
            
            logger.info(f"🤖 INTEGRATED AI DETECTION: {face_count} faces, {person_count} persons, Total: {people_count}, FPS: {self.fps:.1f}")
            
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
ai_detector = IntegratedAIPilgrimDetector()

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
    ai_detector.run_integrated_detection()

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
# Enhanced backend with location and emergency features
import phonenumbers
from twilio.rest import Client
import smtplib


class EmergencyService:
    """Advanced emergency service handler"""
    
    def __init__(self):
        self.emergency_contacts = {
            'police': '+91100',
            'ambulance': '+91108',
            'fire': '+91101',
            'women_helpline': '+91109'
        }
        
    def handle_police_emergency(self, user_data, issue_description, location):
        """Handle police emergency with automated reporting"""
        emergency_data = {
            'type': 'police',
            'user': user_data,
            'description': issue_description,
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'priority': 'high' if 'violence' in issue_description.lower() else 'medium'
        }
        
        # Log emergency
        self.log_emergency(emergency_data)
        
        # Send SMS alert to admin
        self.send_emergency_sms(emergency_data)
        
        # Prepare callback data
        return {
            'success': True,
            'emergency_id': emergency_data['timestamp'],
            'message': 'Police emergency registered. Help is on the way!',
            'contact_number': self.emergency_contacts['police']
        }
    
    def handle_medical_emergency(self, user_data, symptoms, severity, location):
        """Handle medical emergencies with triage system"""
        medical_data = {
            'type': 'medical',
            'user': user_data,
            'symptoms': symptoms,
            'severity': severity,
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'triage_level': self.calculate_triage_level(severity, symptoms)
        }
        
        # Log medical emergency
        self.log_emergency(medical_data)
        
        # Determine appropriate response
        contact_number = self.emergency_contacts['ambulance']
        if severity.lower() in ['critical', 'high']:
            # Immediate ambulance dispatch
            self.dispatch_ambulance(medical_data)
        
        return {
            'success': True,
            'emergency_id': medical_data['timestamp'],
            'message': f'Medical assistance alerted. {severity.upper()} case.',
            'contact_number': contact_number,
            'advice': self.generate_medical_advice(symptoms)
        }
    
    def calculate_triage_level(self, severity, symptoms):
        """Calculate medical triage level"""
        critical_keywords = ['heart', 'stroke', 'unconscious', 'bleeding', 'breathing']
        if severity.lower() == 'critical' or any(keyword in symptoms.lower() for keyword in critical_keywords):
            return 'immediate'
        elif severity.lower() == 'high':
            return 'urgent'
        else:
            return 'standard'
    
    def dispatch_ambulance(self, medical_data):
        """Simulate ambulance dispatch"""
        # In real implementation, integrate with ambulance services
        logger.info(f"🚑 AMBULANCE DISPATCH: {medical_data}")
        
    def send_emergency_sms(self, emergency_data):
        """Send SMS alert (would require Twilio integration)"""
        try:
            # Example with Twilio (commented for simulation)
            # client = Client(twilio_account_sid, twilio_auth_token)
            # message = client.messages.create(
            #     body=f"EMERGENCY: {emergency_data['type']} at {emergency_data['location']}",
            #     from_=twilio_number,
            #     to=admin_number
            # )
            logger.info(f"📱 Emergency SMS simulated: {emergency_data}")
        except Exception as e:
            logger.error(f"SMS sending failed: {e}")
    
    def log_emergency(self, emergency_data):
        """Log emergency to database"""
        conn = sqlite3.connect('pilgrimage.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO emergency_logs 
            (type, user_data, description, location, severity, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            emergency_data['type'],
            json.dumps(emergency_data.get('user', {})),
            emergency_data.get('description', '') or emergency_data.get('symptoms', ''),
            emergency_data.get('location', ''),
            emergency_data.get('severity', 'medium'),
            'active',
            emergency_data['timestamp']
        ))
        
        conn.commit()
        conn.close()

# Enhanced temples with real coordinates (Delhi area examples)
enhanced_temples = [
    {
        'id': 1,
        'name': 'Akshardham Temple',
        'capacity': 200,
        'current_crowd': 45,
        'wait_time': 15,
        'status': 'open',
        'opening_time': '06:00',
        'closing_time': '21:00',
        'location': {
            'latitude': 28.6129,
            'longitude': 77.2779,
            'address': 'Akshardham Temple, Delhi',
            'google_maps_url': 'https://maps.google.com/?q=28.6129,77.2779'
        },
        'contact': '+911123456789',
        'facilities': ['wheelchair', 'medical', 'food']
    },
    {
        'id': 2,
        'name': 'Lotus Temple',
        'capacity': 150,
        'current_crowd': 78,
        'wait_time': 25,
        'status': 'open',
        'opening_time': '07:00',
        'closing_time': '20:00',
        'location': {
            'latitude': 28.5535,
            'longitude': 77.2588,
            'address': 'Lotus Temple, Delhi',
            'google_maps_url': 'https://maps.google.com/?q=28.5535,77.2588'
        },
        'contact': '+911123456790',
        'facilities': ['wheelchair', 'parking', 'meditation']
    },
    {
        'id': 3,
        'name': 'ISKCON Temple',
        'capacity': 100,
        'current_crowd': 32,
        'wait_time': 8,
        'status': 'open',
        'opening_time': '08:00',
        'closing_time': '19:00',
        'location': {
            'latitude': 28.6328,
            'longitude': 77.2197,
            'address': 'ISKCON Temple, Delhi',
            'google_maps_url': 'https://maps.google.com/?q=28.6328,77.2197'
        },
        'contact': '+911123456791',
        'facilities': ['food', 'accommodation', 'library']
    }
]

# New API endpoints
@app.route('/api/temples/<int:temple_id>/navigate', methods=['GET'])
@token_required
def get_temple_navigation(current_user, temple_id):
    """Get temple navigation details"""
    temple = next((t for t in enhanced_temples if t['id'] == temple_id), None)
    
    if not temple:
        return jsonify({'error': 'Temple not found'}), 404
    
    return jsonify({
        'temple_name': temple['name'],
        'navigation': temple['location'],
        'contact': temple['contact'],
        'current_status': temple['status']
    })

@app.route('/api/emergency/police', methods=['POST'])
@token_required
def report_police_emergency(current_user):
    """Report police emergency"""
    data = request.get_json()
    
    emergency_service = EmergencyService()
    result = emergency_service.handle_police_emergency(
        user_data={'username': current_user},
        issue_description=data.get('description', ''),
        location=data.get('location', 'Unknown')
    )
    
    return jsonify(result)

@app.route('/api/emergency/medical', methods=['POST'])
@token_required
def report_medical_emergency(current_user):
    """Report medical emergency"""
    data = request.get_json()
    
    emergency_service = EmergencyService()
    result = emergency_service.handle_medical_emergency(
        user_data={'username': current_user},
        symptoms=data.get('symptoms', ''),
        severity=data.get('severity', 'medium'),
        location=data.get('location', 'Unknown')
    )
    
    return jsonify(result)

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
