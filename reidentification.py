import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logger
logger = logging.getLogger(__name__)

class TrackFeatures:
    """Store features and metadata for a tracked person"""
    def __init__(self, track_id, max_history=10):
        self.track_id = track_id
        self.features = deque(maxlen=max_history)  # Store multiple feature vectors
        self.last_seen = 0  # Frame number when last seen
        self.first_seen = 0  # Frame number when first seen
        self.consecutive_matches = 0  # Count consecutive re-identifications
        self.area_ratio_history = deque(maxlen=max_history)  # Store recent area ratios
        self.position_history = deque(maxlen=max_history)  # Store recent positions (center x,y)
        self.confidence = 0.0  # Re-identification confidence
    
    def add_feature(self, feature):
        """Add a feature vector to history"""
        self.features.append(feature)
    
    def add_area_ratio(self, ratio):
        """Add area ratio to history"""
        self.area_ratio_history.append(ratio)
    
    def add_position(self, box):
        """Add position (center of box) to history"""
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        self.position_history.append((center_x, center_y))
    
    def get_mean_feature(self):
        """Get mean feature vector across history"""
        if not self.features:
            return None
        
        try:
            # Make sure all features have the same length before averaging
            if len(self.features) == 1:
                return self.features[0]
                
            # Check if all feature vectors have the same shape
            first_feature_len = len(self.features[0])
            valid_features = [f for f in self.features if len(f) == first_feature_len]
            
            if valid_features:
                return np.mean(np.array(valid_features), axis=0)
            else:
                # If no valid features, return the most recent one
                return self.features[-1]
        except Exception as e:
            # Fallback in case of any error
            print(f"Warning: Error calculating mean feature: {e}")
            return self.features[-1] if self.features else None
    
    def get_mean_area_ratio(self):
        """Get mean area ratio across history"""
        if not self.area_ratio_history:
            return 0.0
        return np.mean(self.area_ratio_history)

    def get_last_position(self):
        """Get last known position"""
        if not self.position_history:
            return None
        return self.position_history[-1]
    
    def calculate_consistency(self):
        """Calculate track consistency score based on history"""
        if len(self.area_ratio_history) < 2:
            return 0.0
        
        # Calculate variance in area ratios (lower is better)
        area_variance = np.var(self.area_ratio_history)
        
        # Calculate average movement between frames
        if len(self.position_history) < 2:
            avg_movement = 0.0
        else:
            movements = []
            for i in range(1, len(self.position_history)):
                prev = self.position_history[i-1]
                curr = self.position_history[i]
                dist = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                movements.append(dist)
            avg_movement = np.mean(movements) if movements else 0.0
        
        # Combined score: higher is better
        consistency = 1.0 / (1.0 + area_variance + 0.1 * avg_movement)
        return min(consistency, 1.0)  # Cap at 1.0


class PersonReIdentifier:
    """Handle person re-identification across frames"""
    
    def __init__(self, config: Dict[str, Any]):
        # Configuration parameters
        self.feature_method = config.get('reid_feature_method', 'histogram')  # 'histogram' or 'hog'
        self.similarity_threshold = config.get('reid_similarity_threshold', 0.7)
        self.cache_duration = config.get('reid_cache_duration', 100)  # Frames to keep departed IDs in cache
        self.position_weight = config.get('reid_position_weight', 0.3)  # Weight for position vs appearance
        self.area_weight = config.get('reid_area_weight', 0.2)  # Weight for area ratio similarity
        self.min_consecutive_matches = config.get('reid_min_consecutive_matches', 3)
        self.stable_confidence_threshold = config.get('reid_stable_confidence_threshold', 0.8)
        
        # HOG descriptor for feature extraction
        self.hog = cv2.HOGDescriptor()
        
        # Storage for active and historical identities
        self.active_identities = {}  # track_id -> TrackFeatures
        self.historical_identities = {}  # track_id -> TrackFeatures
        
        # Frame counter
        self.frame_counter = 0
        
        # Main user tracking
        self.stable_main_user_id = None
        self.main_user_confidence = 0.0
        
        logger.info("Person ReIdentifier initialized with:")
        logger.info(f"- Feature method: {self.feature_method}")
        logger.info(f"- Similarity threshold: {self.similarity_threshold}")
        logger.info(f"- Cache duration: {self.cache_duration} frames")
    
    def extract_features(self, frame: np.ndarray, box) -> np.ndarray:
        """Extract feature vector from person image region"""
        try:
            # Extract person ROI
            x1, y1, x2, y2 = [int(coord) for coord in box]
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Skip if box is invalid
            if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0:
                return np.zeros(96)  # Return zero feature vector with fixed size
            
            person_roi = frame[y1:y2, x1:x2]
            
            # Skip if ROI is empty
            if person_roi.size == 0:
                return np.zeros(96)
            
            # Standardized histogram features (consistent dimensions)
            # Always use simple RGB histogram for consistency
            hist_features = []
            for channel in range(3):  # For each color channel (RGB)
                hist = cv2.calcHist([person_roi], [channel], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)
            
            # Ensure we always return exactly 96 features (32 bins × 3 channels)
            feature_vector = np.array(hist_features)
            
            if len(feature_vector) != 96:
                logger.warning(f"Feature vector has unexpected size: {len(feature_vector)}")
                # Pad or truncate to ensure consistent length
                if len(feature_vector) < 96:
                    # Pad with zeros if too short
                    return np.pad(feature_vector, (0, 96 - len(feature_vector)))
                else:
                    # Truncate if too long
                    return feature_vector[:96]
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return a zero vector of consistent size
            return np.zeros(96)

    def calculate_position_similarity(self, pos1, pos2, frame_shape) -> float:
        """Calculate position similarity based on normalized distance"""
        if pos1 is None or pos2 is None:
            return 0.0
        
        # Normalize coordinates by frame dimensions
        width, height = frame_shape[1], frame_shape[0]
        norm_pos1 = (pos1[0] / width, pos1[1] / height)
        norm_pos2 = (pos2[0] / width, pos2[1] / height)
        
        # Calculate Euclidean distance
        distance = np.sqrt((norm_pos1[0] - norm_pos2[0])**2 + (norm_pos1[1] - norm_pos2[1])**2)
        
        # Convert to similarity (1 - normalized distance)
        # Maximum normalized distance is sqrt(2), so divide by that
        similarity = 1.0 - (distance / np.sqrt(2))
        return max(0.0, similarity)  # Ensure non-negative

    def calculate_area_ratio_similarity(self, ratio1, ratio2) -> float:
        """Calculate similarity between area ratios"""
        if ratio1 == 0 or ratio2 == 0:
            return 0.0
            
        # Use ratio of the smaller to the larger to get similarity
        similarity = min(ratio1, ratio2) / max(ratio1, ratio2)
        return similarity

    def update_tracks(self, frame: np.ndarray, stable_tracks: List, area_ratios: Dict[int, float]):
        """Update track features with new frame information"""
        self.frame_counter += 1
        
        try:
            # Update active identities with new information
            current_track_ids = set()
            
            for track in stable_tracks:
                if not hasattr(track, 'id'):
                    continue  # Skip tracks without ID
                    
                track_id = track.id
                current_track_ids.add(track_id)
                
                # Get area ratio for this track
                area_ratio = area_ratios.get(track_id, 0.0)
                
                # Safe feature extraction
                try:
                    features = self.extract_features(frame, track.box)
                except Exception as e:
                    logger.error(f"Failed to extract features for track {track_id}: {e}")
                    continue  # Skip this track
                
                # Update or create identity
                if track_id not in self.active_identities:
                    try:
                        # Check if this is a returning identity from historical cache
                        matched_id = self.match_with_historical(frame, track, features, area_ratio)
                        
                        if matched_id is not None and matched_id in self.historical_identities:
                            # Restore from historical identity
                            identity = self.historical_identities.pop(matched_id)
                            logger.info(f"Re-identified historical ID {matched_id} -> {track_id}")
                            # Update with new track_id but keep history
                            identity.track_id = track_id
                        else:
                            # Create new identity
                            identity = TrackFeatures(track_id)
                            identity.first_seen = self.frame_counter
                    except Exception as e:
                        logger.error(f"Error in re-identification matching: {e}")
                        # Fallback to new identity
                        identity = TrackFeatures(track_id)
                        identity.first_seen = self.frame_counter
                    
                    # Store in active identities
                    self.active_identities[track_id] = identity
                
                # Update existing identity
                try:
                    identity = self.active_identities[track_id]
                    identity.last_seen = self.frame_counter
                    identity.add_feature(features)
                    identity.add_area_ratio(area_ratio)
                    identity.add_position(track.box)
                    
                    # Check if this is the most consistent identity (potential main user)
                    consistency = identity.calculate_consistency()
                    identity.confidence = consistency
                    
                    # Count consecutive matches to increase confidence
                    identity.consecutive_matches += 1
                    
                    # Log high-confidence identities
                    if identity.confidence > 0.9:
                        logger.debug(f"High confidence track {track_id}: {identity.confidence:.2f}, " +
                                    f"area: {area_ratio:.1f}%, consecutive: {identity.consecutive_matches}")
                except Exception as e:
                    logger.error(f"Error updating identity {track_id}: {e}")
            
            # Check for disappeared tracks
            disappeared_track_ids = set(self.active_identities.keys()) - current_track_ids
            
            for track_id in disappeared_track_ids:
                # Move to historical identities
                identity = self.active_identities.pop(track_id)
                self.historical_identities[track_id] = identity
                logger.debug(f"Track {track_id} disappeared, moved to historical cache")
            
            # Clean up old historical identities
            self.clean_historical_identities()
            
            # Update main user tracking
            self.update_main_user_tracking()
            
        except Exception as e:
            logger.error(f"Critical error in update_tracks: {e}")
            # Continue execution to avoid crashing the application

    def match_with_historical(self, frame, track, features, area_ratio) -> Optional[int]:
        """Try to match a new track with historical identities"""
        if not self.historical_identities:
            return None
            
        try:
            best_match_id = None
            best_match_similarity = 0.0
            
            current_position = None
            if hasattr(track, 'box'):
                center_x = (track.box[0] + track.box[2]) / 2
                center_y = (track.box[1] + track.box[3]) / 2
                current_position = (center_x, center_y)
            
            # Calculate combined similarity with each historical identity
            for hist_id, hist_identity in self.historical_identities.items():
                # Skip if too old
                frames_gone = self.frame_counter - hist_identity.last_seen
                if frames_gone > self.cache_duration:
                    continue
                    
                # Get features safely
                hist_features = hist_identity.get_mean_feature()
                
                # Skip if either feature set is None
                if hist_features is None or features is None:
                    continue
                
                # Try to calculate appearance similarity safely
                try:
                    # Make sure features have the same shape
                    if len(features) != len(hist_features):
                        # Skip if feature dimensions don't match
                        logger.warning(f"Feature dimension mismatch: {len(features)} vs {len(hist_features)}")
                        continue
                        
                    appearance_similarity = float(cosine_similarity([features], [hist_features])[0][0])
                except Exception as e:
                    logger.warning(f"Error calculating appearance similarity: {e}")
                    appearance_similarity = 0.0
                
                # Calculate position similarity
                hist_position = hist_identity.get_last_position()
                position_similarity = self.calculate_position_similarity(
                    current_position, hist_position, frame.shape)
                
                # Calculate area ratio similarity
                hist_area_ratio = hist_identity.get_mean_area_ratio()
                area_similarity = self.calculate_area_ratio_similarity(area_ratio, hist_area_ratio)
                
                # Combined similarity score
                combined_similarity = (
                    (1 - self.position_weight - self.area_weight) * appearance_similarity +
                    self.position_weight * position_similarity +
                    self.area_weight * area_similarity
                )
                
                # Decay similarity based on time gone
                time_factor = max(0, 1.0 - (frames_gone / self.cache_duration))
                combined_similarity *= time_factor
                
                # Update best match
                if combined_similarity > best_match_similarity and combined_similarity >= self.similarity_threshold:
                    best_match_similarity = combined_similarity
                    best_match_id = hist_id
            
            if best_match_id is not None:
                logger.info(f"Matched with historical ID {best_match_id} (sim={best_match_similarity:.2f})")
                
            return best_match_id
            
        except Exception as e:
            logger.error(f"Error in match_with_historical: {e}")
            return None

    def clean_historical_identities(self):
        """Remove old identities from historical cache"""
        ids_to_remove = []
        
        for track_id, identity in self.historical_identities.items():
            frames_gone = self.frame_counter - identity.last_seen
            if frames_gone > self.cache_duration:
                ids_to_remove.append(track_id)
        
        for track_id in ids_to_remove:
            del self.historical_identities[track_id]
    
    def update_main_user_tracking(self):
        """Update stable main user tracking based on consistency"""
        # Find most consistent identity with sufficient history
        best_id = None
        best_confidence = 0.0
        best_area_ratio = 0.0
        
        for track_id, identity in self.active_identities.items():
            # Only consider identities with enough consecutive matches
            if identity.consecutive_matches >= self.min_consecutive_matches:
                # Consider both confidence and area ratio
                area_ratio = identity.get_mean_area_ratio()
                combined_score = identity.confidence * (0.5 + 0.5 * min(area_ratio / 20.0, 1.0))
                
                if combined_score > best_confidence:
                    best_confidence = combined_score
                    best_id = track_id
                    best_area_ratio = area_ratio
        
        # Update stable main user if confidence is high enough
        if best_id is not None and best_confidence > self.stable_confidence_threshold:
            # Only log if changing
            if best_id != self.stable_main_user_id:
                logger.info(f"Stable main user updated: {best_id} (conf={best_confidence:.2f}, area={best_area_ratio:.1f}%)")
            
            self.stable_main_user_id = best_id
            self.main_user_confidence = best_confidence
        elif best_id is None and self.stable_main_user_id is not None:
            # Main user disappeared completely
            logger.info(f"Stable main user lost: {self.stable_main_user_id}")
            self.stable_main_user_id = None
            self.main_user_confidence = 0.0
    
    def get_stable_main_user(self) -> Tuple[Optional[int], float]:
        """Get the stable main user ID and confidence"""
        return self.stable_main_user_id, self.main_user_confidence
    
    def annotate_frame(self, frame, tracks):
        """Draw re-identification information on frame"""
        for track in tracks:
            if not hasattr(track, 'id') or track.id not in self.active_identities:
                continue
                
            identity = self.active_identities[track.id]
            
            # Draw track box
            box = track.box
            x1, y1, x2, y2 = [int(c) for c in box]
            
            # Determine color based on identity status
            if track.id == self.stable_main_user_id:
                # Stable main user - green
                color = (0, 255, 0)
                thickness = 3
            else:
                # Regular track - blue with opacity based on confidence
                conf = identity.confidence
                color = (255, int(255 * (1-conf)), 0)  # Blue to purple based on confidence
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw track information
            info_text = f"ID:{track.id} C:{identity.confidence:.2f} M:{identity.consecutive_matches}"
            
            # Add "MAIN" label for stable main user
            if track.id == self.stable_main_user_id:
                info_text = "MAIN USER " + info_text
                
            cv2.putText(frame, info_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
