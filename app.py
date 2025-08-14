#!/usr/bin/env python3
"""
GUI App for Human Detection System
Uses the EXACT SAME operations as main.py:
- YOLOHumanDetector with motpy integration
- CustomerGreetingSystem with statistical analysis
- Multi-object tracking
- TTS integration
- Real-time visual results for tuning config
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
import signal
import sys
from datetime import datetime, timedelta
import os
import threading
from typing import Sequence, List, Optional, Dict, Any
import statistics
from collections import deque
import requests
import json
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import queue

# Import motpy components (same as main.py)
from motpy import Detection, MultiObjectTracker, NpImage
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track

# Load configuration from config.json (same as main.py)
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"⚠️ Error loading configuration: {e}")
        print("⚠️ Using default configuration")
        return {}

class YOLOHumanDetector(BaseObjectDetector):
    """YOLO-based human detector that integrates with motpy tracking system - SAME AS main.py"""
    
    def __init__(self, config: Dict[str, Any]):
        super(YOLOHumanDetector, self).__init__()
        
        # Get configuration values with defaults
        model_path = config.get('model_path', 'yolov8n.pt')
        self.conf_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        print(f"🤖 YOLO Human Detector initialized:")
        print(f"   • Model: {model_path}")
        print(f"   • Confidence threshold: {self.conf_threshold}")
        print(f"   • IoU threshold: {self.iou_threshold}")
    
    def process_image(self, image: NpImage) -> Sequence[Detection]:
        """Process image and return human detections in motpy format - SAME AS main.py"""
        # Run YOLO detection with configured parameters
        results = self.model(
            image, 
            verbose=False, 
            conf=self.conf_threshold,
            iou=self.iou_threshold
        )
        
        detections = []
        
        # Extract only human detections (class 0)
        for detection in results[0].boxes:
            if detection.cls == 0:  # Class 0 is person
                box = detection.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax]
                confidence = float(detection.conf.cpu().numpy().item())  # Use item() to safely convert to scalar
                detections.append(Detection(box=box, score=confidence, class_id=0))
        
        return detections

class CustomerGreetingSystem:
    """Comprehensive Customer Greeting System - EXACT SAME AS main.py"""
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize logging
        log_level = config.get('log_level', 'INFO')
        self.logger = setup_logger(__name__, log_level, is_main=True)
        
        # Detection and tracking components
        self.human_detector = YOLOHumanDetector(config)
        self.setup_tracker(config)
        
        # Statistical tracking (from C++ implementation) - SAME AS main.py
        self.detection_ratios = deque(maxlen=1000)  # Store recent detection ratios
        self.human_count = 0
        self.non_human_count = 0
        self.current_human_detected = False
        self.frames_without_human = 0
        self.frame_counter = 0
        
        # Load thresholds from config (with defaults) - SAME AS main.py
        self.HUMAN_COUNT_THRESHOLD = config.get('welcome_threshold', 20)
        self.NON_HUMAN_COUNT_THRESHOLD = config.get('departure_threshold', 18)
        self.MEDIAN_RATIO_ARRIVAL_THRESHOLD = config.get('median_ratio_threshold', 30.0)
        self.MODE_RATIO_ARRIVAL_THRESHOLD = config.get('mode_ratio_threshold', 20.0)
        self.MEDIAN_RATIO_DEPARTURE_THRESHOLD = config.get('departure_median_threshold', 50.0)
        self.MODE_RATIO_DEPARTURE_THRESHOLD = config.get('departure_mode_threshold', 40.0)
        self.CLEAR_RATIOS_FRAME_THRESHOLD = config.get('clear_ratios_after_frames', 30)
        
        # TTS endpoints - SAME AS main.py
        self.tts_endpoints = config.get('tts_endpoints', {
            'welcome': 'https://robot-api2.pvi.digital/api/v1/conversation/tts-hello',
            'departure': 'https://robot-api2.pvi.digital/api/v1/conversation/tts-user-leave'
        })
        
        # Frame management for GUI
        self.latest_frame = None
        self.latest_annotated_frame = None
        
        # Result directory
        self.result_dir = config.get('output_dir', 'result')
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Active tracks for analysis
        self.active_tracks = []
        
        # Enhanced user tracking with object IDs
        self.main_user_id = None  # Track ID of the main user (highest area ratio)
        self.main_user_last_seen = 0  # Frame when main user was last detected
        self.main_user_absence_threshold = config.get('main_user_absence_threshold', 10)  # Frames to wait before considering user departed
        self.track_history = {}  # Track ID -> {last_seen: frame_num, max_area_ratio: float, first_seen: frame_num}
        
        print("🚀 Customer Greeting System initialized with configuration:")
        print(f"   • Welcome threshold: {self.HUMAN_COUNT_THRESHOLD} frames")
        print(f"   • Departure threshold: {self.NON_HUMAN_COUNT_THRESHOLD} frames")
        print(f"   • Arrival median ratio: {self.MEDIAN_RATIO_ARRIVAL_THRESHOLD}%")
        print(f"   • Arrival mode ratio: {self.MODE_RATIO_ARRIVAL_THRESHOLD}%")
        print(f"   • Output directory: {self.result_dir}")
        print(f"   • Log level: {log_level}")
    
    def setup_tracker(self, config: Dict[str, Any]):
        """Setup motpy multi-object tracker - SAME AS main.py"""
        # Default tracker parameters
        model_spec = {
            'order_pos': 1,      # 1st order (constant velocity) for position
            'dim_pos': 2,        # 2D position (x, y)
            'order_size': 0,     # 0th order (constant) for size
            'dim_size': 2,       # 2D size (width, height)
            'q_var_pos': 5000.,  # Process noise for position
            'r_var_pos': 0.1     # Measurement noise for position
        }
        
        # Calculate dt based on video FPS from config
        fps = config.get('video_fps', 30)
        dt = 1 / fps
        
        # Create tracker with configured parameters
        self.tracker = MultiObjectTracker(
            dt=dt, 
            model_spec=model_spec,
            tracker_kwargs={'max_staleness': 12},  # Keep tracks longer for better consistency
            matching_fn_kwargs={'min_iou': config.get('iou_threshold', 0.25)}  # IoU threshold from config
        )
        
        print(f"🎯 Multi-object tracker initialized:")
        print(f"   • FPS: {fps}")
        print(f"   • dt: {dt}")
        print(f"   • IoU threshold: {config.get('iou_threshold', 0.25)}")
        
    def calculate_area_ratio(self, box, frame_shape):
        """Calculate ratio between detection box area and frame area - SAME AS main.py"""
        frame_area = frame_shape[0] * frame_shape[1]  # height * width
        box_area = (box[2] - box[0]) * (box[3] - box[1])  # width * height
        return (box_area / frame_area) * 100.0  # Return as percentage
    
    def calculate_median_ratio(self) -> float:
        """Calculate median of stored detection ratios - SAME AS main.py"""
        if not self.detection_ratios:
            return 0.0
        return statistics.median(self.detection_ratios)
    
    def calculate_mode_ratio(self, tolerance: float = 1.0) -> float:
        """Calculate mode of detection ratios with tolerance - SAME AS main.py"""
        if not self.detection_ratios:
            return 0.0
        
        # Round ratios to nearest tolerance value
        rounded_ratios = [round(ratio / tolerance) * tolerance for ratio in self.detection_ratios]
        
        try:
            return statistics.mode(rounded_ratios)
        except statistics.StatisticsError:
            # If no unique mode, return median
            return statistics.median(rounded_ratios)
    
    def update_track_history(self, stable_tracks):
        """Update track history and identify main user based on area ratios"""
        current_tracks = set()
        
        # Update existing tracks and identify potential main user
        max_area_ratio = 0.0
        potential_main_user = None
        
        for track in stable_tracks:
            track_id = track.id
            current_tracks.add(track_id)
            area_ratio = self.calculate_area_ratio(track.box, self.latest_frame.shape)
            
            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'first_seen': self.frame_counter,
                    'last_seen': self.frame_counter,
                    'max_area_ratio': area_ratio,
                    'area_ratios': deque(maxlen=10)  # Store recent area ratios for this track
                }
            else:
                self.track_history[track_id]['last_seen'] = self.frame_counter
                self.track_history[track_id]['max_area_ratio'] = max(
                    self.track_history[track_id]['max_area_ratio'], 
                    area_ratio
                )
            
            self.track_history[track_id]['area_ratios'].append(area_ratio)
            
            # Check if this could be the main user (highest area ratio)
            if area_ratio > max_area_ratio:
                max_area_ratio = area_ratio
                potential_main_user = track_id
        
        # Identify main user during arrival phase
        if self.main_user_id is None and potential_main_user and max_area_ratio > 15.0:  # Threshold for main user
            self.main_user_id = potential_main_user
            print(f"🎯 Main user identified: Track ID {self.main_user_id} with area ratio {max_area_ratio:.2f}%")
        
        # Update main user last seen if they're still present
        if self.main_user_id in current_tracks:
            self.main_user_last_seen = self.frame_counter
        
        # Clean up old tracks that haven't been seen for a while
        tracks_to_remove = []
        for track_id, history in self.track_history.items():
            if self.frame_counter - history['last_seen'] > 30:  # Remove tracks not seen for 30 frames
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
            if track_id == self.main_user_id:
                print(f"🚫 Main user track {track_id} removed from history")
    
    def check_main_user_departure(self) -> bool:
        """Check if main user has departed based on track absence"""
        if self.main_user_id is None:
            return False
        
        frames_since_last_seen = self.frame_counter - self.main_user_last_seen
        
        # Main user departed if not seen for threshold frames
        if frames_since_last_seen >= self.main_user_absence_threshold:
            print(f"👋 Main user (Track ID {self.main_user_id}) departed after {frames_since_last_seen} frames absence")
            return True
        
        return False
    
    def reset_main_user_tracking(self):
        """Reset main user tracking state"""
        print(f"🔄 Resetting main user tracking (was tracking ID: {self.main_user_id})")
        self.main_user_id = None
        self.main_user_last_seen = 0
        self.track_history.clear()
    
    def save_detection_image(self, frame, prefix: str = "detection"):
        """Save the frame with detection boxes drawn - SAME AS main.py"""
        try:
            # Use the latest annotated frame if available, otherwise annotate current frame
            if self.latest_annotated_frame is not None:
                annotated_frame = self.latest_annotated_frame.copy()
            else:
                annotated_frame = frame.copy()
                # Draw current tracks on frame
                for track in self.active_tracks:
                    draw_track(annotated_frame, track, thickness=2, text_at_bottom=True)
        
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.result_dir, f"{prefix}_{timestamp}.jpg")
            
            # Save the image
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Saved {prefix} image: {filename}")
            return filename
        except Exception as e:
            print(f"⚠️ Failed to save {prefix} image: {e}")
            return None
    
    def make_tts_request(self, endpoint_type: str):
        """Make HTTP POST request to TTS endpoint - SAME AS main.py"""
        try:
            # Get the endpoint URL from config
            if endpoint_type == 'welcome':
                url = self.tts_endpoints.get('welcome')
            elif endpoint_type == 'departure':
                url = self.tts_endpoints.get('departure')
            else:
                print(f"⚠️ Unknown TTS endpoint type: {endpoint_type}")
                return False # Return False for unknown type
            
            if not url:
                print(f"⚠️ No TTS endpoint configured for {endpoint_type}")
                return False # Return False if URL not found
                
            # Make the request with client_id in payload
            payload = {"client_id": "fd9da2874b393784"}
            print(f"🔊 Making TTS request to: {url} with payload: {payload}")
            response = requests.post(url, json=payload, timeout=10) # Increased timeout
            
            if response.status_code == 200:
                print(f"✅ TTS request successful: {endpoint_type}")
                return True
            else:
                print(f"⚠️ TTS request failed: {endpoint_type}, status: {response.status_code}, response: {response.text}")
                return False
        except requests.exceptions.Timeout:
            print(f"⚠️ TTS request timeout: {url}")
            return False
        except Exception as e:
            print(f"⚠️ TTS request error: {e}")
            return False
    
    def process_frame_detections(self, frame):
        """Process frame for human detection and tracking - SAME AS main.py"""
        self.frame_counter += 1
        
        # Update latest frame for GUI
        self.latest_frame = frame
        
        # Get detections from YOLO
        detections = self.human_detector.process_image(frame)
        
        # Update tracker with detections
        self.active_tracks = self.tracker.step(detections)
        
        # Filter for active tracks (minimum 3 steps alive for stability)
        stable_tracks = [track for track in self.active_tracks if hasattr(track, 'id')]
        
        # Update track history and main user identification
        self.update_track_history(stable_tracks)
        
        # Create annotated frame for visualization and saving
        self.latest_annotated_frame = frame.copy()
        
        # Draw detections and tracks
        for det in detections:
            draw_detection(self.latest_annotated_frame, det)
        
        for track in stable_tracks:
            # Highlight main user track in different color
            if hasattr(track, 'id') and track.id == self.main_user_id:
                draw_track(self.latest_annotated_frame, track, thickness=3, text_at_bottom=True)
                # Add "MAIN USER" text
                cv2.putText(self.latest_annotated_frame, "MAIN USER", 
                           (int(track.box[0]), int(track.box[1] - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                draw_track(self.latest_annotated_frame, track, thickness=2, text_at_bottom=True)
        
        # Calculate area ratios for statistical analysis
        has_human = False
        max_area_ratio = 0.0
        
        for track in stable_tracks:
            area_ratio = self.calculate_area_ratio(track.box, frame.shape)
            self.detection_ratios.append(area_ratio)
            max_area_ratio = max(max_area_ratio, area_ratio)
            has_human = True
        
        # Update counters
        if has_human:
            self.human_count += 1
            self.frames_without_human = 0
        else:
            self.non_human_count += 1
            self.frames_without_human += 1
        
        # Clear detection ratios if no human detected for extended period
        if self.frames_without_human >= self.CLEAR_RATIOS_FRAME_THRESHOLD:
            self.detection_ratios.clear()
            self.human_count = 0
            self.non_human_count = 0
            self.frames_without_human = 0
            print("🧹 Cleared detection ratios and counts after 200 frames without human")
        
        # Debug logging every 10 frames
        if self.frame_counter % 10 == 0:
            self.log_debug_info(has_human, max_area_ratio)
        
        # Check for arrival/departure conditions
        self.check_greeting_conditions(frame, has_human)
        
        return detections, stable_tracks
    
    def log_debug_info(self, has_human: bool, max_area_ratio: float):
        """Log debug information - SAME AS main.py"""
        median_ratio = self.calculate_median_ratio()
        mode_ratio = self.calculate_mode_ratio(1.0)
        
        print(f"🔍 Debug [Frame {self.frame_counter}]:")
        print(f"   • Human Count: {self.human_count}")
        print(f"   • Non-Human Count: {self.non_human_count}")
        print(f"   • Current Human Detected Flag: {self.current_human_detected}")
        print(f"   • Median Ratio: {median_ratio:.2f}%")
        print(f"   • Mode Ratio: {mode_ratio:.2f}%")
        print(f"   • Current Frame Has Human: {'yes' if has_human else 'no'}")
        print(f"   • Max Area Ratio: {max_area_ratio:.2f}%")
        print(f"   • Active Tracks: {len(self.active_tracks)}")
        print(f"   • Frames Without Human: {self.frames_without_human}")
        print(f"   • Detection Ratios Size: {len(self.detection_ratios)}")
        
        # Log arrival conditions
        print("   • Arrival Conditions:")
        print(f"     - Count >= {self.HUMAN_COUNT_THRESHOLD}: {'✅' if self.human_count >= self.HUMAN_COUNT_THRESHOLD else '❌'}")
        print(f"     - Not Currently Detected: {'✅' if not self.current_human_detected else '❌'}")
        print(f"     - Median > {self.MEDIAN_RATIO_ARRIVAL_THRESHOLD}%: {'✅' if median_ratio > self.MEDIAN_RATIO_ARRIVAL_THRESHOLD else '❌'}")
        print(f"     - Mode > {self.MODE_RATIO_ARRIVAL_THRESHOLD}%: {'✅' if mode_ratio > self.MODE_RATIO_ARRIVAL_THRESHOLD else '❌'}")
        
        # Log departure conditions
        print("   • Departure Conditions:")
        print(f"     - Non-Human Count > {self.NON_HUMAN_COUNT_THRESHOLD}: {'✅' if self.non_human_count > self.NON_HUMAN_COUNT_THRESHOLD else '❌'}")
        print(f"     - Currently Detected: {'✅' if self.current_human_detected else '❌'}")
        print(f"     - Median < {self.MEDIAN_RATIO_DEPARTURE_THRESHOLD}%: {'✅' if median_ratio < self.MEDIAN_RATIO_DEPARTURE_THRESHOLD else '❌'}")
        print(f"     - Mode < {self.MODE_RATIO_DEPARTURE_THRESHOLD}%: {'✅' if mode_ratio < self.MODE_RATIO_DEPARTURE_THRESHOLD else '❌'}")
    
    def check_greeting_conditions(self, frame, has_human: bool):
        """Check statistical conditions for customer arrival and departure - SAME AS main.py"""
        median_ratio = self.calculate_median_ratio()
        mode_ratio = self.calculate_mode_ratio(1.0)
        
        # ** Customer Arrival Detection **
        if (self.human_count >= self.HUMAN_COUNT_THRESHOLD and 
            not self.current_human_detected and 
            median_ratio > self.MEDIAN_RATIO_ARRIVAL_THRESHOLD and 
            mode_ratio > self.MODE_RATIO_ARRIVAL_THRESHOLD):
            
            print(f"👤 CUSTOMER ARRIVAL DETECTED!")
            print(f"   • Human detected {self.human_count} times")
            print(f"   • Median detection ratio: {median_ratio:.2f}% of frame")
            print(f"   • Mode detection ratio: {mode_ratio:.2f}% of frame")
            print(f"   • Active tracks: {len(self.active_tracks)}")
            print("🎉 Welcome! Nice to see you! 👋")
            
            # Save arrival image
            self.save_detection_image(frame, "customer_arrival")
            
            # Make TTS request for greeting
            self.make_tts_request("welcome")
            
            # Update state
            self.current_human_detected = True
            self.human_count = 0
            self.detection_ratios.clear()
        
        # ** Enhanced Customer Departure Detection **
        # Check main user departure first (primary method)
        main_user_departed = self.check_main_user_departure()
        
        # Traditional statistical departure detection (backup method)
        statistical_departure = (self.non_human_count > self.NON_HUMAN_COUNT_THRESHOLD and 
                                self.current_human_detected and 
                                median_ratio < self.MEDIAN_RATIO_DEPARTURE_THRESHOLD and 
                                mode_ratio < self.MODE_RATIO_DEPARTURE_THRESHOLD)
        
        # Trigger departure if either method detects departure
        if (main_user_departed or statistical_departure) and self.current_human_detected:
            
            departure_method = "Main user tracking" if main_user_departed else "Statistical analysis"
            print(f"👋 CUSTOMER DEPARTURE DETECTED! (Method: {departure_method})")
            print(f"   • No human detected for {self.non_human_count} frames")
            print(f"   • Median detection ratio: {median_ratio:.2f}% of frame")
            print(f"   • Mode detection ratio: {mode_ratio:.2f}% of frame")
            if self.main_user_id:
                frames_since_last_seen = self.frame_counter - self.main_user_last_seen
                print(f"   • Main user (ID: {self.main_user_id}) last seen {frames_since_last_seen} frames ago")
            print("👋 Goodbye! Thanks for visiting!")
            
            # Save departure image
            self.save_detection_image(frame, "customer_departure")
            
            # Make TTS request for farewell
            self.make_tts_request("departure")
            
            # Reset state
            self.current_human_detected = False
            self.non_human_count = 0
            self.human_count = 0
            self.detection_ratios.clear()
            
            # Reset main user tracking
            self.reset_main_user_tracking()

class GUIApp:
    """GUI Application that uses the exact same operations as main.py"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.greeting_system = CustomerGreetingSystem(config)
        
        # Video processing
        self.cap = None
        self.video_writer = None
        self.is_running = False
        self.is_paused = False
        
        # GUI elements
        self.root = None
        self.video_label = None
        self.status_label = None
        self.control_frame = None
        
        # Threading
        self.video_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = deque(maxlen=30)
        
        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🚀 GUI App initialized with CustomerGreetingSystem")
    
    def create_gui(self):
        """Create the main GUI window"""
        self.root = tk.Tk()
        self.root.title("Human Detection System - Same Operations as main.py")
        self.root.geometry("1200x800")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Video display
        self.video_label = ttk.Label(main_frame, text="Initializing video...", 
                                    borderwidth=2, relief="solid")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Status display
        self.status_label = ttk.Label(main_frame, text="Ready", 
                                     font=("Arial", 12, "bold"))
        self.status_label.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Control frame
        self.control_frame = ttk.Frame(main_frame)
        self.control_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Control buttons
        ttk.Button(self.control_frame, text="Start", command=self.start_detection).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Stop", command=self.stop_detection).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Pause/Resume", command=self.toggle_pause).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Save Frame", command=self.save_current_frame).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Settings", command=self.show_settings).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Quit", command=self.quit_app).pack(side="right", padx=5)
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Detection Info (Same as main.py)")
        info_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Detection counters
        self.human_count_label = ttk.Label(info_frame, text="Human Count: 0")
        self.human_count_label.pack(side="left", padx=10)
        
        self.non_human_count_label = ttk.Label(info_frame, text="No Human Count: 0")
        self.non_human_count_label.pack(side="left", padx=10)
        
        self.fps_label = ttk.Label(info_frame, text="FPS: 0.0")
        self.fps_label.pack(side="left", padx=10)
        
        self.status_indicator = ttk.Label(info_frame, text="Status: No Human", 
                                         foreground="red")
        self.status_indicator.pack(side="left", padx=10)
        
        # Statistical info frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistical Analysis (Same as main.py)")
        stats_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        self.median_ratio_label = ttk.Label(stats_frame, text="Median Ratio: 0.0%")
        self.median_ratio_label.pack(side="left", padx=10)
        
        self.mode_ratio_label = ttk.Label(stats_frame, text="Mode Ratio: 0.0%")
        self.mode_ratio_label.pack(side="left", padx=10)
        
        self.tracks_label = ttk.Label(stats_frame, text="Active Tracks: 0")
        self.tracks_label.pack(side="left", padx=10)
        
        self.ratios_size_label = ttk.Label(stats_frame, text="Ratios Size: 0")
        self.ratios_size_label.pack(side="left", padx=10)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        
        print("✅ GUI created successfully")
    
    def start_detection(self):
        """Start the detection process"""
        if not self.is_running:
            try:
                # Initialize video capture
                input_source = self.config.get('input_source', 0)
                self.cap = cv2.VideoCapture(input_source)
                
                if not self.cap.isOpened():
                    messagebox.showerror("Error", f"Could not open video source: {input_source}")
                    return
                
                # Get video properties
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"📹 Video properties: {width}x{height} @ {fps:.1f} FPS")
                
                # Initialize video writer if saving video
                if self.config.get('save_video', False):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = os.path.join(self.greeting_system.result_dir, f"gui_recording_{timestamp}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(video_path, fourcc, self.config.get('video_fps', 30), (width, height))
                    print(f"📹 Video will be saved to: {video_path}")
                
                # Start video processing thread
                self.is_running = True
                self.video_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
                self.video_thread.start()
                
                print("✅ Detection started")
                self.update_status_display("Detection started")
                
            except Exception as e:
                print(f"❌ Error starting detection: {e}")
                messagebox.showerror("Error", f"Failed to start detection: {e}")
    
    def stop_detection(self):
        """Stop the detection process"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print("🛑 Detection stopped")
        self.update_status_display("Detection stopped")
    
    def toggle_pause(self):
        """Toggle pause/resume of detection"""
        self.is_paused = not self.is_paused
        status = "paused" if self.is_paused else "resumed"
        print(f"Detection {status}")
        self.update_status_display(f"Detection {status}")
    
    def save_current_frame(self):
        """Save the current frame"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.greeting_system.result_dir, f"manual_save_{timestamp}.jpg")
            cv2.imwrite(filename, self.current_frame)
            print(f"📸 Manually saved frame: {filename}")
            self.update_status_display(f"Frame saved: {filename}")
    
    def show_settings(self):
        """Show settings dialog for tuning config"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings - Tune Detection Parameters")
        settings_window.geometry("500x600")
        
        # Add settings
        ttk.Label(settings_window, text="Detection Parameters", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Welcome threshold
        ttk.Label(settings_window, text="Welcome Threshold (frames):").pack()
        welcome_var = tk.IntVar(value=self.greeting_system.HUMAN_COUNT_THRESHOLD)
        welcome_scale = ttk.Scale(settings_window, from_=5, to=100, variable=welcome_var, orient="horizontal")
        welcome_scale.pack(fill="x", padx=20)
        
        # Departure threshold
        ttk.Label(settings_window, text="Departure Threshold (frames):").pack()
        departure_var = tk.IntVar(value=self.greeting_system.NON_HUMAN_COUNT_THRESHOLD)
        departure_scale = ttk.Scale(settings_window, from_=5, to=100, variable=departure_var, orient="horizontal")
        departure_scale.pack(fill="x", padx=20)
        
        # Median ratio threshold
        ttk.Label(settings_window, text="Arrival Median Ratio Threshold (%):").pack()
        median_var = tk.DoubleVar(value=self.greeting_system.MEDIAN_RATIO_ARRIVAL_THRESHOLD)
        median_scale = ttk.Scale(settings_window, from_=5.0, to=80.0, variable=median_var, orient="horizontal")
        median_scale.pack(fill="x", padx=20)
        
        # Mode ratio threshold
        ttk.Label(settings_window, text="Arrival Mode Ratio Threshold (%):").pack()
        mode_var = tk.DoubleVar(value=self.greeting_system.MODE_RATIO_ARRIVAL_THRESHOLD)
        mode_scale = ttk.Scale(settings_window, from_=5.0, to=80.0, variable=mode_var, orient="horizontal")
        mode_scale.pack(fill="x", padx=20)
        
        # Save settings button
        def save_settings():
            self.greeting_system.HUMAN_COUNT_THRESHOLD = welcome_var.get()
            self.greeting_system.NON_HUMAN_COUNT_THRESHOLD = departure_var.get()
            self.greeting_system.MEDIAN_RATIO_ARRIVAL_THRESHOLD = median_var.get()
            self.greeting_system.MODE_RATIO_ARRIVAL_THRESHOLD = mode_var.get()
            print(f"✅ Settings updated: welcome={welcome_var.get()}, departure={departure_var.get()}, median={median_var.get()}, mode={mode_var.get()}")
            settings_window.destroy()
        
        ttk.Button(settings_window, text="Save Settings", command=save_settings).pack(pady=10)
    
    def quit_app(self):
        """Quit the application"""
        self.stop_detection()
        if self.root:
            self.root.quit()
        print("👋 Application quitting")
    
    def video_processing_loop(self):
        """Main video processing loop using the exact same operations as main.py"""
        print("🚀 Starting video processing loop with CustomerGreetingSystem")
        
        try:
            while self.is_running and self.cap and self.cap.isOpened():
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Failed to capture image")
                    break
                
                # Store current frame for manual save
                self.current_frame = frame.copy()
                
                # Process every Nth frame based on detection interval
                detection_interval = self.config.get('detection_interval', 1)
                if self.frame_count % detection_interval == 0:
                    # Use raw BGR frame for processing (same as main.py)
                    # Do NOT convert to RGB here; only convert for GUI display
                    detections, tracks = self.greeting_system.process_frame_detections(frame)
                
                # Save video frame if enabled
                if self.video_writer and self.greeting_system.latest_annotated_frame is not None:
                    self.video_writer.write(self.greeting_system.latest_annotated_frame)
                
                # Update GUI display (convert BGR->RGB for GUI only)
                self.update_gui_display()
                
                # Update counters in GUI
                if self.root:
                    self.root.after(0, self.update_gui_counters)
                
                self.frame_count += 1
                
                # Control frame rate
                time.sleep(0.01)  
                
        except Exception as e:
            print(f"❌ Error in video processing loop: {e}")
        finally:
            print("✅ Video processing loop ended")
    
    def update_gui_display(self):
        """Update the GUI video display"""
        try:
            # Use the annotated frame from the greeting system (BGR)
            if self.greeting_system.latest_annotated_frame is not None:
                frame_bgr = self.greeting_system.latest_annotated_frame
            else:
                frame_bgr = self.greeting_system.latest_frame
                
            if frame_bgr is not None:
                # Resize frame for display
                display_frame_bgr = cv2.resize(frame_bgr, (800, 600))
                
                # Convert BGR to RGB for GUI display
                display_frame_rgb = cv2.cvtColor(display_frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(display_frame_rgb)
                
                # Convert to PhotoImage
                photo_image = ImageTk.PhotoImage(pil_image)
                
                # Update GUI in main thread
                if self.root:
                    self.root.after(0, lambda: self.video_label.config(image=photo_image))
                    # Keep a reference to prevent garbage collection
                    self.video_label.image = photo_image
                    
        except Exception as e:
            print(f"❌ Error updating GUI display: {e}")
    
    def update_gui_counters(self):
        """Update GUI counter displays with real-time data from greeting system"""
        try:
            if hasattr(self, 'human_count_label'):
                self.human_count_label.config(text=f"Human Count: {self.greeting_system.human_count}")
            if hasattr(self, 'non_human_count_label'):
                self.non_human_count_label.config(text=f"No Human Count: {self.greeting_system.non_human_count}")
            if hasattr(self, 'fps_label'):
                fps = self.calculate_fps()
                self.fps_label.config(text=f"FPS: {fps:.1f}")
            if hasattr(self, 'status_indicator'):
                if self.greeting_system.current_human_detected:
                    self.status_indicator.config(text="Status: Human Detected", foreground="green")
                else:
                    self.status_indicator.config(text="Status: No Human", foreground="red")
            
            # Update statistical information
            if hasattr(self, 'median_ratio_label'):
                median = self.greeting_system.calculate_median_ratio()
                self.median_ratio_label.config(text=f"Median Ratio: {median:.2f}%")
            if hasattr(self, 'mode_ratio_label'):
                mode = self.greeting_system.calculate_mode_ratio(1.0)
                self.mode_ratio_label.config(text=f"Mode Ratio: {mode:.2f}%")
            if hasattr(self, 'tracks_label'):
                self.tracks_label.config(text=f"Active Tracks: {len(self.greeting_system.active_tracks)}")
            if hasattr(self, 'ratios_size_label'):
                self.ratios_size_label.config(text=f"Ratios Size: {len(self.greeting_system.detection_ratios)}")
                
        except Exception as e:
            print(f"❌ Error updating GUI counters: {e}")
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 0:
            fps = self.frame_count / elapsed
            self.fps_history.append(fps)
            return fps
        return 0.0
    
    def update_status_display(self, message):
        """Update the status display in GUI"""
        if self.status_label:
            current_time = datetime.now().strftime("%H:%M:%S")
            status_text = f"[{current_time}] {message}"
            self.status_label.config(text=status_text)
    
    def signal_handler(self, signum, frame):
        """Handle termination signals"""
        print(f"🛑 Received signal {signum}. Shutting down gracefully...")
        self.is_running = False
        if self.root:
            self.root.after(0, self.quit_app)
    
    def run(self):
        """Run the GUI application"""
        print("🚀 Starting GUI App with CustomerGreetingSystem...")
        print(f"Input source: {self.config.get('input_source', 0)}")
        print(f"Model: {self.config.get('model_path', 'yolov8n.pt')}")
        
        # Create GUI
        self.create_gui()
        
        # Start GUI main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("⌨️ Keyboard interrupt received")
        except Exception as e:
            print(f"❌ Error in GUI main loop: {e}")
        finally:
            # Cleanup
            print("🧹 Cleaning up resources...")
            self.stop_detection()
            print("✅ GUI App stopped")

def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='GUI Human Detection System - Same Operations as main.py')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--output', type=str, default='result', help='Output directory')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--no-save-images', action='store_true', help='Disable saving detection images')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"⚠️ Error loading configuration: {e}")
            return
    
    # Override config with command line arguments
    if config is None:
        config = {}
    
    config['model_path'] = args.model
    config['input_source'] = int(args.source) if args.source.isdigit() else args.source
    config['output_dir'] = args.output
    config['log_level'] = args.log_level
    config['save_video'] = args.save_video
    config['save_images'] = not args.no_save_images
    
    print(f"🚀 Configuration:")
    print(f"   • Input source: {config['input_source']}")
    print(f"   • Model: {config['model_path']}")
    print(f"   • Output directory: {config['output_dir']}")
    print(f"   • Using EXACT SAME operations as main.py")
    
    # Create and run GUI app
    try:
        app = GUIApp(config)
        app.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
