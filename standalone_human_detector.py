#!/usr/bin/env python3
"""
Standalone Human Detection System
A comprehensive application that combines YOLO inference, human detection logic, and logging.
This system detects human appearance and departure using the same logic as the C++ implementation.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
import os
import time
import json
import requests
from datetime import datetime
from collections import deque
import argparse
import signal
import sys
from pathlib import Path

class HumanDetector:
    """Main human detection class with comprehensive logging and state management"""
    
    @staticmethod
    def _get_default_config():
        """Get default configuration"""
        return {
            'model_path': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'input_source': 0,  # 0 for webcam, or path to video file
            'output_dir': 'output',
            'log_dir': 'logs',
            'save_images': True,
            'save_video': False,
            'video_fps': 30,
            'detection_interval': 1,  # Process every Nth frame
            'headless': True,  # Run without GUI display
            'welcome_threshold': 60,  # Frames with human to trigger welcome
            'departure_threshold': 80,  # Frames without human to trigger departure
            'median_ratio_threshold': 30.0,  # % of frame area for welcome
            'mode_ratio_threshold': 25.0,    # % of frame area for welcome
            'departure_median_threshold': 50.0,  # % of frame area for departure
            'departure_mode_threshold': 40.0,    # % of frame area for departure
            'clear_ratios_after_frames': 200,    # Clear ratios after N frames without human
            'tts_endpoints': {
                'welcome': 'https://robot-api2.pvi.digital/api/v1/conversation/tts-hello',
                'departure': 'https://robot-api2.pvi.digital/api/v1/conversation/tts-user-leave'
            },
            'log_level': 'INFO'
        }
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.setup_logging()
        self.setup_device()
        self.load_model()
        self.setup_directories()
        self.initialize_detection_state()
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = deque(maxlen=30)
        
        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = True
        logger.info("HumanDetector initialized successfully")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        global logger
        
        # Create log directory
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config['log_dir'], f"human_detector_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file),
                logging.FileHandler(os.path.join(self.config['log_dir'], 'latest.log'))
            ]
        )
        
        logger = logging.getLogger('HumanDetector')
        logger.info(f"Logging initialized. Log file: {log_file}")
    
    def setup_device(self):
        """Setup GPU/CPU device for inference"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
            
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
            logger.info("GPU memory fraction set to 80%")
        else:
            logger.info("Running on CPU - consider using GPU for better performance")
    
    def load_model(self):
        """Load YOLO model on appropriate device"""
        try:
            logger.info(f"Loading YOLO model: {self.config['model_path']}")
            self.model = YOLO(self.config['model_path'])
            
            if self.device == 'cuda':
                self.model.to(self.device)
                logger.info("YOLO model loaded successfully on GPU")
            else:
                logger.info("YOLO model loaded successfully on CPU")
                
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def setup_directories(self):
        """Create necessary output directories"""
        directories = [
            self.config['output_dir'],
            self.config['log_dir'],
            os.path.join(self.config['output_dir'], 'detections'),
            os.path.join(self.config['output_dir'], 'departures')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def initialize_detection_state(self):
        """Initialize detection state variables"""
        self.human_count = 0
        self.non_human_count = 0
        self.current_human_detected = False
        self.detection_ratios = []
        self.last_detection_time = None
        self.detection_history = deque(maxlen=1000)
        
        logger.info("Detection state initialized")
    
    def calculate_median_ratio(self):
        """Calculate median of detection ratios"""
        if not self.detection_ratios:
            return 0.0
        
        sorted_ratios = sorted(self.detection_ratios)
        size = len(sorted_ratios)
        if size % 2 == 0:
            return (sorted_ratios[size//2 - 1] + sorted_ratios[size//2]) / 2.0
        else:
            return sorted_ratios[size//2]
    
    def calculate_mode_ratio(self, tolerance=1.0):
        """Calculate mode of detection ratios with tolerance"""
        if not self.detection_ratios:
            return 0.0
        
        frequency = {}
        
        # Round ratios to nearest integer considering tolerance
        for ratio in self.detection_ratios:
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
    
    def save_detection_image(self, frame, prefix, detections=None):
        """Save detection image with optional bounding boxes"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if prefix == 'detection':
                filename = os.path.join(self.config['output_dir'], 'detections', f"human_{timestamp}.jpg")
            else:
                filename = os.path.join(self.config['output_dir'], 'departures', f"human_left_{timestamp}.jpg")
            
            # Draw bounding boxes if detections provided
            if detections is not None:
                annotated_frame = frame.copy()
                for det in detections:
                    if int(det[5]) == 0:  # Class 0 is person
                        x1, y1, x2, y2, conf, _ = det
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Draw box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"Person: {conf:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                frame_to_save = annotated_frame
            else:
                frame_to_save = frame
            
            # Convert RGB to BGR for OpenCV
            if len(frame_to_save.shape) == 3 and frame_to_save.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_to_save
            
            cv2.imwrite(filename, frame_bgr)
            logger.info(f"Saved {prefix} image: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving detection image: {e}")
            return None
    
    def make_tts_request(self, endpoint):
        """Make synchronous HTTP POST request to TTS endpoint"""
        try:
            payload = {"client_id": "fd9da2874b393784"}
            response = requests.post(endpoint, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"TTS request successful: {endpoint}")
                return True
            else:
                logger.warning(f"TTS request failed with status {response.status_code}: {endpoint}")
                return False
        except requests.exceptions.Timeout:
            logger.error(f"TTS request timeout: {endpoint}")
            return False
        except Exception as e:
            logger.error(f"Error making TTS request to {endpoint}: {e}")
            return False
    
    def process_frame(self, frame):
        """Process single frame for human detection"""
        try:
            # Run YOLO detection
            results = self.model(frame, verbose=False, device=self.device, 
                               conf=self.config['confidence_threshold'], 
                               iou=self.config['iou_threshold'])
            
            detections = results[0].boxes.data.cpu().numpy()
            
            # Check for human detections (class 0 is person in YOLO)
            has_human = False
            frame_area = frame.shape[0] * frame.shape[1]
            human_detections = []
            
            for det in detections:
                if int(det[5]) == 0 and det[4] > self.config['confidence_threshold']:
                    has_human = True
                    human_detections.append(det)
                    
                    # Calculate detection box area ratio
                    x1, y1, x2, y2 = map(int, det[:4])
                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = (box_area / frame_area) * 100.0
                    
                    self.detection_ratios.append(area_ratio)
                    
                    # Store detection in history
                    detection_info = {
                        'timestamp': datetime.now().isoformat(),
                        'frame': self.frame_count,
                        'area_ratio': area_ratio,
                        'confidence': float(det[4]),
                        'bbox': [x1, y1, x2, y2]
                    }
                    self.detection_history.append(detection_info)
            
            # Update counters
            if has_human:
                self.human_count += 1
                self.non_human_count = 0
            else:
                self.non_human_count += 1
            
            # Calculate statistics
            median_ratio = self.calculate_median_ratio()
            mode_ratio = self.calculate_mode_ratio(1.0)
            
            # Log detection status every 10 frames
            if self.frame_count % 10 == 0:
                self.log_detection_status(has_human, median_ratio, mode_ratio, human_detections)
            
            # Check for human appearance (welcome)
            if (self.human_count >= self.config['welcome_threshold'] and 
                not self.current_human_detected and 
                median_ratio > self.config['median_ratio_threshold'] and 
                mode_ratio > self.config['mode_ratio_threshold']):
                
                self.handle_human_appearance(frame, human_detections, median_ratio, mode_ratio)
            
            # Check for human departure
            elif (self.non_human_count > self.config['departure_threshold'] and 
                  self.current_human_detected and 
                  median_ratio < self.config['departure_median_threshold'] and 
                  mode_ratio < self.config['departure_mode_threshold']):
                
                self.handle_human_departure(frame, median_ratio, mode_ratio)
            
            # Clear detection ratios if no human detected for specified frames
            if not has_human and self.non_human_count >= self.config['clear_ratios_after_frames']:
                self.clear_detection_state()
                logger.info(f"🧹 Cleared detection ratios and counts after {self.config['clear_ratios_after_frames']} frames without human")
            
            return has_human, human_detections
            
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            return False, []
    
    def log_detection_status(self, has_human, median_ratio, mode_ratio, detections):
        """Log comprehensive detection status"""
        logger.info(f"Detection Status [Frame {self.frame_count}]:")
        logger.info(f"  • Human Count: {self.human_count}")
        logger.info(f"  • Non-Human Count: {self.non_human_count}")
        logger.info(f"  • Current Human Detected: {self.current_human_detected}")
        logger.info(f"  • Median Ratio: {median_ratio:.2f}%")
        logger.info(f"  • Mode Ratio: {mode_ratio:.2f}%")
        logger.info(f"  • Current Frame Has Human: {has_human}")
        logger.info(f"  • Detection Ratios Size: {len(self.detection_ratios)}")
        logger.info(f"  • Human Detections: {len(detections)}")
        
        if detections:
            for i, det in enumerate(detections):
                conf = det[4]
                x1, y1, x2, y2 = map(int, det[:4])
                area = (x2 - x1) * (y2 - y1)
                frame_area = 640 * 480  # Assuming standard resolution
                ratio = (area / frame_area) * 100
                logger.info(f"    Detection {i+1}: conf={conf:.2f}, ratio={ratio:.2f}%, bbox=({x1},{y1},{x2},{y2})")
    
    def handle_human_appearance(self, frame, detections, median_ratio, mode_ratio):
        """Handle human appearance event"""
        logger.info("👤 Human detected - Welcome!")
        logger.info(f"  • Human Count: {self.human_count}")
        logger.info(f"  • Median Ratio: {median_ratio:.2f}%")
        logger.info(f"  • Mode Ratio: {mode_ratio:.2f}%")
        
        # Save detection image
        if self.config['save_images']:
            self.save_detection_image(frame, 'detection', detections)
        
        # Update state
        self.current_human_detected = True
        self.human_count = 0
        self.last_detection_time = datetime.now()
        
        # Clear detection ratios
        self.detection_ratios.clear()
        
        # Make TTS request synchronously
        self.make_tts_request(self.config['tts_endpoints']['welcome'])
    
    def handle_human_departure(self, frame, median_ratio, mode_ratio):
        """Handle human departure event"""
        logger.info("👋 Human departed - Goodbye!")
        logger.info(f"  • Non-Human Count: {self.non_human_count}")
        logger.info(f"  • Median Ratio: {median_ratio:.2f}%")
        logger.info(f"  • Mode Ratio: {mode_ratio:.2f}%")
        
        # Save departure image
        if self.config['save_images']:
            self.save_detection_image(frame, 'departure')
        
        # Update state
        self.current_human_detected = False
        self.non_human_count = 0
        self.human_count = 0
        
        # Clear detection ratios
        self.detection_ratios.clear()
        
        # Make TTS request synchronously
        self.make_tts_request(self.config['tts_endpoints']['departure'])
    
    def clear_detection_state(self):
        """Clear detection state and counters"""
        self.detection_ratios.clear()
        self.human_count = 0
        self.non_human_count = 0
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 0:
            fps = self.frame_count / elapsed
            self.fps_history.append(fps)
            return fps
        return 0.0
    
    def get_average_fps(self):
        """Get average FPS over recent frames"""
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0
    
    def log_performance_stats(self):
        """Log performance statistics"""
        current_fps = self.calculate_fps()
        avg_fps = self.get_average_fps()
        
        logger.info(f"Performance Stats [Frame {self.frame_count}]:")
        logger.info(f"  • Current FPS: {current_fps:.2f}")
        logger.info(f"  • Average FPS: {avg_fps:.2f}")
        logger.info(f"  • Runtime: {time.time() - self.start_time:.1f}s")
        
        if self.device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"  • GPU Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
    
    def save_detection_summary(self):
        """Save detection summary to JSON file"""
        try:
            summary = {
                'session_info': {
                    'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_frames': self.frame_count,
                    'runtime_seconds': time.time() - self.start_time,
                    'average_fps': self.get_average_fps()
                },
                'detection_stats': {
                    'total_detections': len(self.detection_history),
                    'human_appearances': len([d for d in self.detection_history if 'welcome' in str(d)]),
                    'human_departures': len([d for d in self.detection_history if 'departure' in str(d)])
                },
                'device_info': {
                    'device': self.device,
                    'model_path': self.config['model_path']
                }
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(self.config['log_dir'], f"detection_summary_{timestamp}.json")
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Detection summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving detection summary: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def run(self):
        """Main detection loop"""
        logger.info("Starting human detection system...")
        logger.info(f"Input source: {self.config['input_source']}")
        logger.info(f"Model: {self.config['model_path']}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Headless mode: {self.config['headless']}")
        logger.info("Press Ctrl+C to stop the application")
        
        # Initialize video capture
        if isinstance(self.config['input_source'], int):
            cap = cv2.VideoCapture(self.config['input_source'])
        else:
            cap = cv2.VideoCapture(self.config['input_source'])
        
        if not cap.isOpened():
            logger.error(f"Could not open video source: {self.config['input_source']}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps:.1f} FPS")
        
        # Initialize video writer if saving video
        video_writer = None
        if self.config['save_video']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.config['output_dir'], f"detection_video_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, self.config['video_fps'], (width, height))
            logger.info(f"Video will be saved to: {video_path}")
        
        logger.info("Entering main detection loop...")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("End of video stream or failed to read frame")
                    break
                
                # Process every Nth frame based on detection interval
                if self.frame_count % self.config['detection_interval'] == 0:
                    # Convert BGR to RGB for YOLO
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    has_human, detections = self.process_frame(frame_rgb)
                    
                    # Draw detection boxes on frame for video output
                    if detections:
                        for det in detections:
                            x1, y1, x2, y2, conf, _ = det
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"Person: {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add status text to frame for video output
                    status_text = f"Human: {self.human_count} | No Human: {self.non_human_count} | FPS: {self.get_average_fps():.1f}"
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Log performance every 100 frames
                    if self.frame_count % 100 == 0:
                        self.log_performance_stats()
                        logger.info(f"Processed {self.frame_count} frames...")
                        
                        # Clear GPU cache if using CUDA
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                
                # Save video frame if enabled
                if video_writer:
                    video_writer.write(frame)
                
                # Remove GUI display - this is a headless application
                # cv2.imshow('Human Detection', frame)
                
                # Handle key presses - removed since no GUI
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord('q'):
                #     logger.info("Quit key pressed")
                #     break
                # elif key == ord('s'):
                #     # Save current frame
                #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                #     filename = os.path.join(self.config['output_dir'], f"manual_save_{timestamp}.jpg")
                #     cv2.imwrite(filename, frame)
                #     logger.info(f"Manually saved frame: {filename}")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            # Cleanup
            logger.info("Cleaning up resources...")
            cap.release()
            if video_writer:
                video_writer.release()
            # cv2.destroyAllWindows()  # Remove since no GUI windows
            
            # Save final summary
            self.save_detection_summary()
            
            logger.info("Human detection system stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Standalone Human Detection System')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--no-save-images', action='store_true', help='Disable saving detection images')
    parser.add_argument('--gui', action='store_true', help='Enable GUI display (not recommended for headless servers)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return
    
    # Override config with command line arguments
    if config is None:
        config = {}
    
    # Ensure all required config keys exist
    default_config = HumanDetector._get_default_config()
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    config['model_path'] = args.model
    config['input_source'] = int(args.source) if args.source.isdigit() else args.source
    config['output_dir'] = args.output
    config['log_level'] = args.log_level
    config['save_video'] = args.save_video
    config['save_images'] = not args.no_save_images
    config['headless'] = not args.gui  # Default to headless unless --gui is specified
    
    print(f"Configuration: Headless mode = {config['headless']}")
    print(f"Input source: {config['input_source']}")
    print(f"Model: {config['model_path']}")
    print(f"Output directory: {config['output_dir']}")
    
    # Create and run detector
    try:
        detector = HumanDetector(config)
        detector.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
