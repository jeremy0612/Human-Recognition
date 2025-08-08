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
from aiortc import VideoStreamTrack
from webrtc_server import init_server
from aiohttp import web
import threading

class HumanDetector:
    def __init__(self):
        self.human_present = False
        self.human_start_time = None
        self.greeting_triggered = False
        self.AREA_RATIO_THRESHOLD = 1/7  # Area ratio threshold
        self.TIME_THRESHOLD = 3  # Time threshold in seconds
        self.latest_frame = None
        self.frame_ready = threading.Event()
        
        # Create result directory if it doesn't exist
        self.result_dir = "result"
        os.makedirs(self.result_dir, exist_ok=True)
        
    def calculate_area_ratio(self, box, frame_shape):
        """Calculate ratio between detection box area and frame area"""
        frame_area = frame_shape[0] * frame_shape[1]  # height * width
        box_area = (box[2] - box[0]) * (box[3] - box[1])  # width * height
        return box_area / frame_area
    
    def save_detection_image(self, frame, results):
        """Save the frame with detection boxes drawn"""
        # Draw detection boxes on the frame
        annotated_frame = results[0].plot()
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.result_dir, f"detection_{timestamp}.jpg")
        
        # Save the image
        cv2.imwrite(filename, annotated_frame)
        print(f"\nSaved detection image: {filename}")
    
    def check_human_presence(self, results, frame):
        """Check for human presence and handle greeting logic"""
        current_time = datetime.now()
        human_detected = False
        max_area_ratio = 0
        
        # Update latest frame for WebRTC
        self.latest_frame = frame
        self.frame_ready.set()
        
        # Check all detections for humans
        for detection in results[0].boxes:
            if detection.cls == 0:  # Class 0 is person in YOLO
                human_detected = True
                box = detection.xyxy[0].cpu().numpy()  # Get box coordinates
                area_ratio = self.calculate_area_ratio(box, frame.shape)
                max_area_ratio = max(max_area_ratio, area_ratio)
        
        # Update human presence tracking
        if human_detected and max_area_ratio >= self.AREA_RATIO_THRESHOLD:
            if not self.human_present:
                self.human_present = True
                self.human_start_time = current_time
                print(f"\nHuman detected! Area ratio: {max_area_ratio:.3f}")
            elif not self.greeting_triggered:
                time_present = (current_time - self.human_start_time).total_seconds()
                if time_present >= self.TIME_THRESHOLD:
                    print("\n🎉 Welcome! Nice to see you! 👋")
                    self.greeting_triggered = True
                    # Save the frame when greeting is triggered
                    self.save_detection_image(frame, results)
        else:
            if self.human_present:
                print("\nHuman no longer detected or too small")
            self.human_present = False
            self.human_start_time = None
            self.greeting_triggered = False

class VideoStream(VideoStreamTrack):
    def __init__(self, human_detector):
        super().__init__()
        self.human_detector = human_detector
        self.frame_count = 0

    async def recv(self):
        self.frame_count += 1
        # Wait for a new frame
        self.human_detector.frame_ready.wait()
        self.human_detector.frame_ready.clear()
        
        # Convert frame to RGB
        frame = cv2.cvtColor(self.human_detector.latest_frame, cv2.COLOR_BGR2RGB)
        
        # Create VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = 1/30  # 30 fps
        
        return video_frame

def run_detection(human_detector):
    # Load the YOLO model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Webcam: {frame_width}x{frame_height} at {fps} FPS")
    print("\nPress Ctrl+C to stop the application\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Perform YOLO detection
            results = model(frame, verbose=False)
            
            # Check for human presence and handle greeting
            human_detector.check_human_presence(results, frame)
            
    except KeyboardInterrupt:
        print("\nStopping application...")
    
    finally:
        cap.release()
        print("Application closed")

async def run_server(human_detector):
    app = init_server()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8088)
    await site.start()
    print("WebRTC server running at http://localhost:8088")

def main():
    # Initialize human detector
    human_detector = HumanDetector()
    
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Start detection in a separate thread
    detection_thread = threading.Thread(target=run_detection, args=(human_detector,))
    detection_thread.start()
    
    try:
        # Run WebRTC server in the main thread
        loop.run_until_complete(run_server(human_detector))
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nStopping application...")
    finally:
        loop.close()

if __name__ == "__main__":
    main()
