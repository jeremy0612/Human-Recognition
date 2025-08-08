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
import fractions  # Add import for fractions

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

def signal_handler(sig, frame):
    """Handle Ctrl+C signal to gracefully stop the client"""
    global running
    logger.info("Received interrupt signal, shutting down...")
    running = False

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

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
        
        # Create VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count
        # Fix: Use fractions.Fraction instead of float for time_base
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
            "http://localhost:8080/offer",
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
        logger.info("Streaming video from camera. Press Ctrl+C to stop.")
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