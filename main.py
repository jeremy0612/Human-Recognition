import cv2
import numpy as np
import time
from ultralytics import YOLO
import signal
import sys
from datetime import datetime, timedelta
import os
import asyncio
from av import VideoFrame
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription
from webrtc_server import init_server
from aiohttp import web
import threading
from typing import Sequence, List, Optional, Dict, Any
import statistics
from collections import deque
import requests
import json
import logging

# Import motpy components
from motpy import Detection, MultiObjectTracker, NpImage
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track

# Load configuration from config.json
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
    """YOLO-based human detector that integrates with motpy tracking system"""
    
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
        """Process image and return human detections in motpy format"""
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
    """Comprehensive Customer Greeting System with YOLO detection, Motpy tracking, and statistical analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize logging
        log_level = config.get('log_level', 'INFO')
        self.logger = setup_logger(__name__, log_level, is_main=True)
        
        # Detection and tracking components
        self.human_detector = YOLOHumanDetector(config)
        self.setup_tracker(config)
        
        # Statistical tracking (from C++ implementation)
        self.detection_ratios = deque(maxlen=1000)  # Store recent detection ratios
        self.human_count = 0
        self.non_human_count = 0
        self.current_human_detected = False
        self.frames_without_human = 0
        self.frame_counter = 0
        
        # Load thresholds from config (with defaults)
        self.HUMAN_COUNT_THRESHOLD = config.get('welcome_threshold', 20)
        self.NON_HUMAN_COUNT_THRESHOLD = config.get('departure_threshold', 18)
        self.MEDIAN_RATIO_ARRIVAL_THRESHOLD = config.get('median_ratio_threshold', 30.0)
        self.MODE_RATIO_ARRIVAL_THRESHOLD = config.get('mode_ratio_threshold', 20.0)
        self.MEDIAN_RATIO_DEPARTURE_THRESHOLD = config.get('departure_median_threshold', 50.0)
        self.MODE_RATIO_DEPARTURE_THRESHOLD = config.get('departure_mode_threshold', 40.0)
        self.CLEAR_RATIOS_FRAME_THRESHOLD = config.get('clear_ratios_after_frames', 30)
        
        # TTS endpoints
        self.tts_endpoints = config.get('tts_endpoints', {
            'welcome': 'https://robot-api2.pvi.digital/api/v1/conversation/tts-hello',
            'departure': 'https://robot-api2.pvi.digital/api/v1/conversation/tts-user-leave'
        })
        
        # WebRTC and frame management
        self.latest_frame = None
        self.frame_ready = threading.Event()
        self.latest_annotated_frame = None
        
        # Result directory
        self.result_dir = config.get('output_dir', 'result')
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Active tracks for analysis
        self.active_tracks = []
        
        print("🚀 Customer Greeting System initialized with configuration:")
        print(f"   • Welcome threshold: {self.HUMAN_COUNT_THRESHOLD} frames")
        print(f"   • Departure threshold: {self.NON_HUMAN_COUNT_THRESHOLD} frames")
        print(f"   • Arrival median ratio: {self.MEDIAN_RATIO_ARRIVAL_THRESHOLD}%")
        print(f"   • Arrival mode ratio: {self.MODE_RATIO_ARRIVAL_THRESHOLD}%")
        print(f"   • Output directory: {self.result_dir}")
        print(f"   • Log level: {log_level}")
    
    def setup_tracker(self, config: Dict[str, Any]):
        """Setup motpy multi-object tracker with parameters from config"""
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
        """Calculate ratio between detection box area and frame area (as percentage)"""
        frame_area = frame_shape[0] * frame_shape[1]  # height * width
        box_area = (box[2] - box[0]) * (box[3] - box[1])  # width * height
        return (box_area / frame_area) * 100.0  # Return as percentage
    
    def calculate_median_ratio(self) -> float:
        """Calculate median of stored detection ratios"""
        if not self.detection_ratios:
            return 0.0
        return statistics.median(self.detection_ratios)
    
    def calculate_mode_ratio(self, tolerance: float = 1.0) -> float:
        """Calculate mode of detection ratios with tolerance"""
        if not self.detection_ratios:
            return 0.0
        
        # Round ratios to nearest tolerance value
        rounded_ratios = [round(ratio / tolerance) * tolerance for ratio in self.detection_ratios]
        
        try:
            return statistics.mode(rounded_ratios)
        except statistics.StatisticsError:
            # If no unique mode, return median
            return statistics.median(rounded_ratios)
    
    def save_detection_image(self, frame, prefix: str = "detection"):
        """Save the frame with detection boxes drawn"""
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
        """Make HTTP POST request to TTS endpoint"""
        try:
            # Get the endpoint URL from config
            if endpoint_type == 'welcome':
                url = self.tts_endpoints.get('welcome')
            elif endpoint_type == 'departure':
                url = self.tts_endpoints.get('departure')
            else:
                print(f"⚠️ Unknown TTS endpoint type: {endpoint_type}")
                return
            
            if not url:
                print(f"⚠️ No TTS endpoint configured for {endpoint_type}")
                return
                
            # Make the request
            print(f"🔊 Making TTS request to: {url}")
            response = requests.post(url, json={}, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ TTS request successful: {endpoint_type}")
            else:
                print(f"⚠️ TTS request failed: {endpoint_type}, status: {response.status_code}")
        except Exception as e:
            print(f"⚠️ TTS request error: {e}")
    
    def process_frame_detections(self, frame):
        """Process frame for human detection and tracking"""
        self.frame_counter += 1
        
        # Update latest frame for WebRTC
        self.latest_frame = frame
        self.frame_ready.set()
        
        # Get detections from YOLO
        detections = self.human_detector.process_image(frame)
        
        # Update tracker with detections
        self.active_tracks = self.tracker.step(detections)
        
        # Filter for active tracks (minimum 3 steps alive for stability)
        stable_tracks = [track for track in self.active_tracks if hasattr(track, 'id')]
        
        # Create annotated frame for visualization and saving
        self.latest_annotated_frame = frame.copy()
        
        # Draw detections and tracks
        for det in detections:
            draw_detection(self.latest_annotated_frame, det)
        
        for track in stable_tracks:
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
        """Log debug information"""
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
        """Check statistical conditions for customer arrival and departure"""
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
        
        # ** Customer Departure Detection **
        elif (self.non_human_count > self.NON_HUMAN_COUNT_THRESHOLD and 
              self.current_human_detected and 
              median_ratio < self.MEDIAN_RATIO_DEPARTURE_THRESHOLD and 
              mode_ratio < self.MODE_RATIO_DEPARTURE_THRESHOLD):
            
            print(f"👋 CUSTOMER DEPARTURE DETECTED!")
            print(f"   • No human detected for {self.non_human_count} frames")
            print(f"   • Median detection ratio: {median_ratio:.2f}% of frame")
            print(f"   • Mode detection ratio: {mode_ratio:.2f}% of frame")
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

class VideoStream(VideoStreamTrack):
    """WebRTC video stream that provides the latest annotated frame from the greeting system"""
    
    def __init__(self, greeting_system: CustomerGreetingSystem):
        super().__init__()
        self.greeting_system = greeting_system
        self.frame_count = 0

    async def recv(self):
        self.frame_count += 1
        # Wait for a new frame
        self.greeting_system.frame_ready.wait()
        self.greeting_system.frame_ready.clear()
        
        # Use annotated frame if available, otherwise use latest frame
        if self.greeting_system.latest_annotated_frame is not None:
            frame = self.greeting_system.latest_annotated_frame
        else:
            frame = self.greeting_system.latest_frame
        
        if frame is None:
            # Return a black frame if no frame is available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create VideoFrame
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = 1/30  # 30 fps
        
        return video_frame

def run_detection(greeting_system: CustomerGreetingSystem, config: Dict[str, Any]):
    """Main detection loop using the comprehensive greeting system"""
    print("🚀 Starting Customer Greeting System detection loop...")
    
    # Get input source from config
    input_source = config.get('input_source', 0)  # Default to camera 0
    
    # Open webcam or video source
    print(f"📹 Opening video source: {input_source}...")
    cap = cv2.VideoCapture(input_source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {input_source}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video source: {frame_width}x{frame_height} at {fps} FPS")
    print("⌨️  Press Ctrl+C to stop the application")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Detection interval (process every Nth frame)
    detection_interval = config.get('detection_interval', 1)
    frame_count = 0
    
    # Video recording if enabled
    save_video = config.get('save_video', False)
    video_writer = None
    
    if save_video:
        # Create video writer
        output_path = os.path.join(greeting_system.result_dir, 
                                  f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            config.get('video_fps', 30), 
            (frame_width, frame_height)
        )
        print(f"📹 Recording video to: {output_path}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture image")
                break
            
            frame_count += 1
            
            # Process only every Nth frame based on detection_interval
            if frame_count % detection_interval == 0:
                # Process frame through the comprehensive greeting system
                detections, tracks = greeting_system.process_frame_detections(frame)
            
            # Save video frame if enabled
            if save_video and greeting_system.latest_annotated_frame is not None:
                video_writer.write(greeting_system.latest_annotated_frame)
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)  # 10ms delay
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping Customer Greeting System...")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        print("✅ Detection system closed successfully")

async def run_server(greeting_system: CustomerGreetingSystem):
    """Run WebRTC server with the greeting system"""
    try:
        # Initialize server
        app = init_server()
        
        async def stream_handler(request):
            """Handle WebRTC video stream requests"""
            try:
                # Create a new VideoStream with our greeting system
                video_stream = VideoStream(greeting_system)
                
                # Create a peer connection
                pc = RTCPeerConnection()
                pc.addTrack(video_stream)
                
                # Handle the offer/answer exchange
                offer = await request.json()
                await pc.setRemoteDescription(RTCSessionDescription(
                    sdp=offer["sdp"], type=offer["type"]
                ))
                
                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                return web.json_response({
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type
                })
            except Exception as e:
                print(f"⚠️ Stream error: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        # Add route to serve VideoStream
        app.router.add_get('/stream', stream_handler)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 8088)
        await site.start()
        print("🌐 WebRTC server running at http://localhost:8088")
        print("   • Stream endpoint: http://localhost:8088/stream")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"❌ WebRTC server error: {e}")
        import traceback
        traceback.print_exc()

def signal_handler(signum, frame):
    """Handle Ctrl+C signal gracefully"""
    print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
    sys.exit(0)

def main():
    """Main function to initialize and run the Customer Greeting System"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Initializing Customer Greeting System...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create log directory if it doesn't exist (before setting up logging)
        log_dir = config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        log_level = config.get('log_level', 'INFO')
        log_file_path = os.path.join(log_dir, 'greeting_system.log')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file_path)
            ]
        )
        
        print(f"📝 Logging to: {log_file_path}")
        
        # Initialize the comprehensive greeting system with config
        greeting_system = CustomerGreetingSystem(config)
        
        # Create event loop for WebRTC server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Start detection in a separate thread
        print("🧵 Starting detection thread...")
        detection_thread = threading.Thread(
            target=run_detection, 
            args=(greeting_system, config),
            daemon=True  # Make thread daemon so it exits when main exits
        )
        detection_thread.start()
        
        # Give detection thread time to initialize
        time.sleep(2)
        
        print("🌐 Starting WebRTC server...")
        try:
            # Run WebRTC server in the main thread
            loop.run_until_complete(run_server(greeting_system))
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal...")
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("✅ Customer Greeting System shutdown complete")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
