import asyncio
import json
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, MediaStreamTrack
from av import VideoFrame
import aiohttp
import numpy as np
import logging
import signal
import sys
import fractions
import os
from datetime import datetime
from ultralytics import YOLO
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('webrtc_client.log')
    ]
)
logger = logging.getLogger('webrtc_client')

# Flag to control the main loop
running = True

# Global detection variables
human_count = 0
non_human_count = 0
current_human_detected = False
detection_ratios = []
ratios_mutex = asyncio.Lock()

# Check GPU availability and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

if device == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Set memory fraction to avoid OOM errors
    torch.cuda.set_per_process_memory_fraction(0.8)

# Load YOLO model
try:
    model = YOLO('yolov8n.pt')
    if device == 'cuda':
        model.to(device)
        logger.info("YOLO model loaded successfully on GPU")
    else:
        logger.info("YOLO model loaded successfully on CPU")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    raise

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def signal_handler(sig, frame):
    """Handle Ctrl+C signal to gracefully stop the client"""
    global running
    logger.info("Received interrupt signal, shutting down...")
    running = False

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def calculate_median_ratio():
    """Calculate median of detection ratios"""
    if not detection_ratios:
        return 0.0
    
    sorted_ratios = sorted(detection_ratios)
    size = len(sorted_ratios)
    if size % 2 == 0:
        return (sorted_ratios[size//2 - 1] + sorted_ratios[size//2]) / 2.0
    else:
        return sorted_ratios[size//2]

def calculate_mode_ratio(tolerance=1.0):
    """Calculate mode of detection ratios with tolerance"""
    if not detection_ratios:
        return 0.0
    
    frequency = {}
    
    # Round ratios to nearest integer considering tolerance
    for ratio in detection_ratios:
        rounded = int(ratio / tolerance) * tolerance
        frequency[rounded] = frequency.get(rounded, 0) + 1
    
    # Find the most frequent value
    max_count = 0
    mode = 0.0
    
    for value, count in frequency.items():
        if count > max_count:
            max_count = count
            mode = value
    
    return mode

async def make_tts_request(endpoint):
    """Make HTTP POST request to TTS endpoint"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json={}) as response:
                if response.status == 200:
                    logger.info(f"TTS request successful: {endpoint}")
                else:
                    logger.warning(f"TTS request failed with status {response.status}: {endpoint}")
    except Exception as e:
        logger.error(f"Error making TTS request to {endpoint}: {e}")

async def save_detection_image(frame, prefix):
    """Save the frame with timestamp"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{prefix}_{timestamp}.jpg")
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, frame_bgr)
        logger.info(f"Saved detection image: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving detection image: {e}")
        return False

async def process_human_detection(frame):
    """Process frame for human detection and update counters"""
    global human_count, non_human_count, current_human_detected, detection_ratios
    
    try:
        # Run YOLO detection
        results = model(frame, verbose=False, device=device)
        detections = results[0].boxes.data.cpu().numpy()
        
        # Check for human detections (class 0 is person in YOLO)
        has_human = False
        frame_area = frame.shape[0] * frame.shape[1]
        
        for det in detections:
            if int(det[5]) == 0 and det[4] > 0.5:  # Class 0 (person) with confidence > 0.5
                has_human = True
                # Calculate detection box area ratio
                x1, y1, x2, y2 = map(int, det[:4])
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = (box_area / frame_area) * 100.0
                
                async with ratios_mutex:
                    detection_ratios.append(area_ratio)
                
                break
        
        # Update counters
        if has_human:
            human_count += 1
        else:
            non_human_count += 1
        
        # Calculate statistics
        median_ratio = calculate_median_ratio()
        mode_ratio = calculate_mode_ratio(1.0)
        
        # Log detection status every 10 frames
        if (human_count + non_human_count) % 10 == 0:
            logger.info(f"Detection Status [Frame {human_count + non_human_count}]:")
            logger.info(f"  • Human Count: {human_count}")
            logger.info(f"  • Non-Human Count: {non_human_count}")
            logger.info(f"  • Current Human Detected: {current_human_detected}")
            logger.info(f"  • Median Ratio: {median_ratio:.2f}%")
            logger.info(f"  • Mode Ratio: {mode_ratio:.2f}%")
            logger.info(f"  • Current Frame Has Human: {has_human}")
            logger.info(f"  • Detection Ratios Size: {len(detection_ratios)}")
        
        # Check for human appearance (welcome)
        if (human_count >= 60 and not current_human_detected and 
            median_ratio > 30 and mode_ratio > 25):
            
            logger.info("👤 Human detected - Welcome!")
            logger.info(f"  • Human Count: {human_count}")
            logger.info(f"  • Median Ratio: {median_ratio:.2f}%")
            logger.info(f"  • Mode Ratio: {mode_ratio:.2f}%")
            
            # Make TTS request
            await make_tts_request('https://robot-api2.pvi.digital/api/v1/conversation/tts-hello')
            
            # Save detection image
            await save_detection_image(frame, "human")
            
            # Update state
            current_human_detected = True
            human_count = 0
            
            # Clear detection ratios
            async with ratios_mutex:
                detection_ratios.clear()
        
        # Check for human departure
        elif (non_human_count > 80 and current_human_detected and 
              median_ratio < 50 and mode_ratio < 40):
            
            logger.info("👋 Human departed - Goodbye!")
            logger.info(f"  • Non-Human Count: {non_human_count}")
            logger.info(f"  • Median Ratio: {median_ratio:.2f}%")
            logger.info(f"  • Mode Ratio: {mode_ratio:.2f}%")
            
            # Make TTS request
            await make_tts_request('https://robot-api2.pvi.digital/api/v1/conversation/tts-user-leave')
            
            # Save departure image
            await save_detection_image(frame, "human_left")
            
            # Update state
            current_human_detected = False
            non_human_count = 0
            human_count = 0
            
            # Clear detection ratios
            async with ratios_mutex:
                detection_ratios.clear()
        
        # Clear detection ratios if no human detected for 200 frames
        if not has_human and non_human_count >= 200:
            async with ratios_mutex:
                detection_ratios.clear()
                human_count = 0
                non_human_count = 0
            logger.info("🧹 Cleared detection ratios and counts after 200 frames without human")
        
        return has_human
        
    except Exception as e:
        logger.error(f"Error in human detection: {e}")
        return False

class CameraStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        logger.info("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        self.frame_count = 0
        
        if not self.cap.isOpened():
            logger.error("Could not open video capture device")
            raise Exception("Could not open video capture device")
        logger.info("Camera initialized successfully")

    async def recv(self):
        self.frame_count += 1
        ret, frame = self.cap.read()
        
        if not ret:
            logger.error("Could not read frame from video capture device")
            raise Exception("Could not read frame from video capture device")

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Log every 100th frame
        if self.frame_count % 100 == 0:
            logger.info(f"Captured frame {self.frame_count}, shape: {frame.shape}")
            if device == 'cuda':
                logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated")
        
        # Process human detection
        await process_human_detection(frame)
        
        # Create VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, 30)  # 30 fps
        
        return video_frame

    def stop(self):
        logger.info("Stopping camera capture...")
        self.cap.release()

class VideoReceiver(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0
        logger.info("VideoReceiver initialized")

    async def recv(self):
        try:
            self.frame_count += 1
            frame = await self.track.recv()
            
            # Log every 100th frame
            if self.frame_count % 100 == 0:
                logger.info(f"Received frame {self.frame_count}")
            
            return frame
        except Exception as e:
            logger.error(f"Error receiving frame: {e}")
            raise

async def run_client():
    global running
    # Create peer connection
    pc = RTCPeerConnection()
    logger.info("Created peer connection")
    
    # Create video track from camera
    video_track = CameraStreamTrack()
    pc.addTrack(video_track)
    logger.info("Added camera track to peer connection")

    # Handle received tracks
    @pc.on("track")
    def on_track(track):
        logger.info(f"Received track of kind: {track.kind}")
        if track.kind == "video":
            # We're not displaying the video, but we still need to receive it
            # to keep the connection alive and process server responses
            receiver = VideoReceiver(track)
            
            async def process_video():
                while running:
                    try:
                        await receiver.recv()
                        await asyncio.sleep(0.01)  # Small delay to prevent CPU overload
                    except Exception as e:
                        logger.error(f"Error in process_video: {e}")
                        break
            
            asyncio.ensure_future(process_video())
    
    # Create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    logger.info("Created and set local description")
    
    # Send offer to server
    async with aiohttp.ClientSession() as session:
        logger.info("Sending offer to server...")
        async with session.post(
            "http://192.168.1.122:8090/offer",
            json={
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            },
        ) as response:
            logger.info("Received answer from server")
            answer = await response.json()
            answer = RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            
            # Set remote description
            await pc.setRemoteDescription(answer)
            logger.info("Set remote description")
    
    # Keep connection alive
    try:
        logger.info("Streaming video from camera with human detection. Press Ctrl+C to stop.")
        while running:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        video_track.stop()
        await pc.close()
        logger.info("Cleanup complete")

if __name__ == "__main__":
    asyncio.run(run_client()) 