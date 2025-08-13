import asyncio
import json
import cv2
import numpy as np
from aiohttp import web
from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from ultralytics import YOLO
import logging
import os
from datetime import datetime
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('webrtc_server.log')
    ]
)
logger = logging.getLogger('webrtc_server')

# Check GPU availability and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

if device == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Set memory fraction to avoid OOM errors
    torch.cuda.set_per_process_memory_fraction(0.8)

# Global list to keep track of peer connections
pcs = set()

# Load YOLO model on GPU if available
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

# Create result directory if it doesn't exist
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0
        self.detection_count = 0
        logger.info("VideoTransformTrack initialized")

    def save_detection_image(self, frame, detections):
        """Save the frame with detection boxes drawn if humans are detected"""
        try:
            # Check if any humans (person class) are detected
            has_human = any(int(det[5]) == 0 for det in detections)
            
            if has_human and self.frame_count % 30 == 0:  # Save every 30th frame with humans
                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(result_dir, f"detection_{timestamp}.jpg")
                
                # Draw bounding boxes on a copy of the frame
                annotated_frame = frame.copy()
                for det in detections:
                    if int(det[5]) == 0:  # Class 0 is person in YOLO
                        x1, y1, x2, y2, score, _ = det
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Save the image
                cv2.imwrite(filename, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved detection image: {filename}")
                return True
        except Exception as e:
            logger.error(f"Error saving detection image: {e}")
        return False

    async def recv(self):
        try:
            self.frame_count += 1
            frame = await self.track.recv()
            
            # Convert frame to numpy array for YOLO processing
            img = frame.to_ndarray(format="bgr24")
            
            # Log every 100th frame
            if self.frame_count % 100 == 0:
                logger.info(f"Processing frame {self.frame_count}, shape: {img.shape}")
                if device == 'cuda':
                    logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated, {torch.cuda.memory_reserved() / 1024**2:.1f} MB reserved")
            
            # Run YOLO detection with GPU acceleration
            try:
                results = model(img, verbose=False, device=device)
                
                # Get detections
                detections = results[0].boxes.data.cpu().numpy()
                
                # Log detections
                if len(detections) > 0:
                    self.detection_count += 1
                    classes = [int(det[5]) for det in detections]
                    class_names = [results[0].names[cls_id] for cls_id in classes]
                    logger.info(f"Frame {self.frame_count}: Detected {len(detections)} objects: {class_names}")
                    
                    # Save detection image if humans are present
                    self.save_detection_image(img, detections)
                
                # Clear GPU cache periodically to prevent memory buildup
                if device == 'cuda' and self.frame_count % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error in YOLO inference: {e}")
                # If GPU inference fails, fall back to CPU
                if device == 'cuda':
                    logger.info("Falling back to CPU inference")
                    try:
                        results = model(img, verbose=False, device='cpu')
                        detections = results[0].boxes.data.cpu().numpy()
                        if len(detections) > 0:
                            self.detection_count += 1
                            classes = [int(det[5]) for det in detections]
                            class_names = [results[0].names[cls_id] for cls_id in classes]
                            logger.info(f"Frame {self.frame_count}: CPU fallback - Detected {len(detections)} objects: {class_names}")
                            self.save_detection_image(img, detections)
                    except Exception as cpu_e:
                        logger.error(f"CPU inference also failed: {cpu_e}")
                        detections = np.array([])
                else:
                    detections = np.array([])
            
            # Process the frame for return (no need to draw on it since we're not displaying)
            # Just pass it through as is
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            return new_frame
        except Exception as e:
            logger.error(f"Error in recv: {e}")
            return frame

async def index(request):
    content = open('index.html', 'r').read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open('client.js', 'r').read()
    return web.Response(content_type="application/javascript", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params["sdp"],
        type=params["type"]
    )

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # Handle incoming track
    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "video":
            transformed_track = VideoTransformTrack(track)
            pc.addTrack(transformed_track)
            logger.info("Added transformed video track")

        @track.on("ended")
        async def on_ended():
            logger.info("Track ended")

    # Set the remote description
    await pc.setRemoteDescription(offer)
    logger.info("Set remote description")

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info("Created and set local description")

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )

async def on_shutdown(app):
    # Close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    logger.info("Server shutting down")

def init_app():
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    return app

if __name__ == "__main__":
    app = init_app()
    logger.info("Starting WebRTC server")
    web.run_app(app, host="0.0.0.0", port=8080) 