# Enhanced Robot Greeting System

This system provides a computer vision-based robot greeting system with robust user tracking and re-identification capability.

## Key Features

- **YOLO Object Detection**: Detects humans in camera feed with high accuracy
- **Motpy Tracking**: Multi-object tracking with unique IDs
- **Visual Re-identification**: Maintains identity even through temporary occlusions or movements
- **Statistical Analysis**: Uses median/mode of area ratios for stability
- **WebRTC Streaming**: Provides a web-based visualization interface
- **TTS Integration**: Triggers welcome/farewell speech output

## Enhancements for Stable User Tracking

The system includes enhanced re-identification capabilities to solve common problems:
- **Rapid movements**: When users gesture or move quickly
- **Brief occlusions**: When users are temporarily hidden
- **Frame exits/re-entries**: When users briefly step out of frame
- **Background people**: Distinguishes main user from background people

## Configuration

Key parameters in `config.json`:

```json
{
  "main_user_absence_threshold": 50,    // Frames before considering main user departed
  "reid_feature_method": "histogram",   // Feature extraction method (histogram or hog)
  "reid_similarity_threshold": 0.7,     // Minimum similarity to match identities
  "reid_cache_duration": 120,           // How long to remember disappeared people
  "reid_position_weight": 0.3,          // Weight for position in similarity calculation
  "reid_area_weight": 0.2,              // Weight for area ratio in similarity calculation
  "reid_min_consecutive_matches": 3,     // Minimum consecutive matches for stable tracking
  "reid_stable_confidence_threshold": 0.75, // Confidence threshold for main user
  "reid_cooldown_period": 30,           // Cooldown frames after departure
}
```

## Running the System

```bash
python main.py
```

The system will:
1. Detect people using YOLO
2. Track them with motpy
3. Extract visual features for each person
4. Maintain identity through appearance matching
5. Identify stable main user with highest confidence
6. Trigger welcome/farewell based on stable tracking

## Re-identification Process

The re-identification module provides enhanced stability by:

1. **Feature Extraction**: Extracts visual features from each detected person
2. **Identity Cache**: Maintains history of recently seen people
3. **Similarity Matching**: Matches new detections with historical identities
4. **Confidence Scoring**: Builds confidence through consecutive matches
5. **Position Tracking**: Considers spatial position for better matching
6. **Cooldown Periods**: Prevents rapid welcome/farewell cycles

This ensures the robot can reliably track the main user even through rapid movements, brief occlusions, or when they temporarily leave and re-enter the frame.