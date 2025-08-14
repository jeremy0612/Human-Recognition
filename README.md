# Customer Greeting System

A comprehensive AI-powered customer greeting system that combines YOLO object detection, Motpy multi-object tracking, and statistical analysis to intelligently detect customer arrivals and departures.

## 🌟 Features

### Core Functionality
- **YOLO Human Detection**: Uses YOLOv8n for accurate human detection
- **Motpy Multi-Object Tracking**: Consistent tracking with unique IDs for better customer behavior analysis
- **Statistical Analysis**: Advanced statistical conditions (median/mode ratios) for reliable arrival/departure detection
- **WebRTC Streaming**: Real-time video streaming with detection visualization
- **TTS Integration**: Automatic text-to-speech greetings and farewells
- **Image Logging**: Automatic saving of customer arrival/departure moments

### Key Components
1. **YOLOHumanDetector**: YOLO-based detector integrated with motpy framework
2. **CustomerGreetingSystem**: Main system combining detection, tracking, and statistical analysis
3. **WebRTC Integration**: Real-time streaming with annotated frames
4. **Statistical Engine**: Robust arrival/departure detection using area ratios

This Python implementation maintains full compatibility with the C++ version:
- ✅ Same statistical thresholds
- ✅ Same area ratio calculations  
- ✅ Same arrival/departure logic
- ✅ Same TTS integration
- ✅ Enhanced with consistent tracking IDs
- ✅ Simple WebRTC protocol implementation for further cloud oriented deployment.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Webcam Feed   │───▶│  YOLO Detection  │───▶│ Motpy Tracking  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ TTS Greetings   │◀───│ Statistical      │◀───│ Area Ratio      │
│                 │    │ Analysis         │    │ Calculation     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │ WebRTC Streaming │
                       │ & Image Logging  │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam connected to the system
- Conda environment (recommended)

### Installation
Both usual pip install or conda env are available, there might be few dependencies conflict.
```bash
# Clone the repository
cd human_recognition

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (if not present)
# The system will automatically download yolov8n.pt on first run
```

### Running the System
```bash
# Run the complete greeting system
python main.py
# Assuming that the webRTC server is currently running
```


## 📁 File Structure

```
human_recognition/
├── main.py                          # Main greeting system
├── webrtc_server.py                 # WebRTC server (legacy)
├── webrtc_client.py                 # WebRTC client (legacy)
├── requirements.txt                 # Dependencies
├── motpy/                           # Motpy tracking library
│   ├── examples/
│   │   ├── webcam_face_tracking.py  # Face tracking example
│   │   └── detect_and_track_in_video.py
│   └── motpy/                       # Core motpy modules
├── result/                          # Saved detection images
└── README.md                        # This file
```

## 🔧 Configuration
A `config.json` file is provided to centralize all configuration parameters for the system. This template includes:

- **Motpy Tracker Settings**: Parameters such as `dt`, `max_staleness`, `min_iou`, and Kalman filter options, allowing you to fine-tune the multi-object tracking behavior.
- **Statistical Conditions**: Thresholds for customer arrival and departure detection, including values like `HUMAN_COUNT_THRESHOLD`, `MEDIAN_RATIO_ARRIVAL_THRESHOLD`, `MODE_RATIO_ARRIVAL_THRESHOLD`, and their departure counterparts.
- **Other System Settings**: You can also specify result directories, TTS API endpoints, and frame thresholds.

## 📈 Performance

- **Detection Speed**: ~15-30 FPS (depending on hardware)
- **Memory Usage**: ~500MB (with YOLO model loaded)
- **CPU Usage**: Moderate (optimized with frame delays)
- **Accuracy**: High (YOLO + statistical validation)

## 📄 License

This project inherits the license from the motpy library components. Others used Apache-2.0 license. 

