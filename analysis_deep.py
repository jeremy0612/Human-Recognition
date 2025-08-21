import statistics
import time
from typing import List, Tuple, Dict, Deque, Any
from collections import deque

# ==== CONFIGURATION ====
DEFAULT_CONFIG = {
    'arrival_median_threshold': 30.0,
    'arrival_mode_threshold': 20.0,
    'departure_median_threshold': 50.0,
    'departure_mode_threshold': 40.0,
    'min_detection_frames': 20,
    'min_non_detection_frames': 18,
    'max_history_size': 1000,
    'arrival_time_seconds': 2.0,
    'departure_time_seconds': 3.0,
    'mode_tolerance': 1.0,
    'fps': 30
}


# ==== UTILITY FUNCTIONS ====

def calculate_box_area_ratio(box: List[float], frame_shape: Tuple[int, int]) -> float:
    """
    Tính tỉ lệ diện tích của box so với toàn bộ khung hình
    
    Args:
        box: [x_min, y_min, x_max, y_max] - tọa độ bounding box
        frame_shape: (height, width) - kích thước khung hình
    
    Returns:
        float: Tỉ lệ phần trăm diện tích (0-100)
    """
    frame_height, frame_width = frame_shape
    frame_area = frame_width * frame_height
    
    box_width = box[2] - box[0]  # x_max - x_min
    box_height = box[3] - box[1]  # y_max - y_min
    box_area = box_width * box_height
    
    return (box_area / frame_area) * 100.0


def calculate_mean_ratio(detection_ratios: Deque[float]) -> float:
    """
    Tính giá trị trung bình của các tỉ lệ detection
    
    Args:
        detection_ratios: deque chứa các tỉ lệ detection gần nhất
    
    Returns:
        float: Giá trị trung bình (0 nếu deque rỗng)
    """
    if not detection_ratios:
        return 0.0
    return sum(detection_ratios) / len(detection_ratios)


def calculate_median_ratio(detection_ratios: Deque[float]) -> float:
    """
    Tính giá trị trung vị (median) của các tỉ lệ detection
    
    Args:
        detection_ratios: deque chứa các tỉ lệ detection gần nhất
    
    Returns:
        float: Giá trị trung vị (0 nếu deque rỗng)
    """
    if not detection_ratios:
        return 0.0
    
    # Chuyển deque sang list và sắp xếp
    sorted_ratios = sorted(detection_ratios)
    n = len(sorted_ratios)
    
    # Tính median
    if n % 2 == 1:
        # Số phần tử lẻ: lấy phần tử ở giữa
        return sorted_ratios[n // 2]
    else:
        # Số phần tử chẵn: lấy trung bình của 2 phần tử giữa
        mid = n // 2
        return (sorted_ratios[mid - 1] + sorted_ratios[mid]) / 2


def calculate_mode_ratio(detection_ratios: Deque[float], tolerance: float = 1.0) -> float:
    """
    Tính mode (giá trị xuất hiện nhiều nhất) của các tỉ lệ detection
    
    Args:
        detection_ratios: deque chứa các tỉ lệ detection gần nhất
        tolerance: Độ chính xác khi làm tròn (default: 1.0)
    
    Returns:
        float: Giá trị mode (hoặc median nếu không có mode duy nhất)
    """
    if not detection_ratios:
        return 0.0
    
    # Làm tròn các giá trị theo tolerance để nhóm chúng lại
    rounded_ratios = [round(ratio / tolerance) * tolerance for ratio in detection_ratios]
    
    try:
        return statistics.mode(rounded_ratios)
    except statistics.StatisticsError:
        # Nếu không có mode duy nhất, trả về median
        return calculate_median_ratio(deque(rounded_ratios))


def calculate_all_statistics(detection_ratios: Deque[float], tolerance: float = 1.0) -> Dict[str, float]:
    """
    Tính tất cả các thống kê: mean, median, mode
    
    Args:
        detection_ratios: deque chứa các tỉ lệ detection
        tolerance: Độ chính xác cho mode calculation
    
    Returns:
        Dict: Các giá trị thống kê
    """
    return {
        "mean": calculate_mean_ratio(detection_ratios),
        "median": calculate_median_ratio(detection_ratios),
        "mode": calculate_mode_ratio(detection_ratios, tolerance)
    }


def determine_presence_time(
    detection_ratios: Deque[float],
    fps: int = 30,
    arrival_threshold: float = 30.0,
    departure_threshold: float = 50.0
) -> Dict[str, Any]:
    """
    Xác định thời gian cần để nhận diện người tới/rời dựa trên thống kê
    
    Args:
        detection_ratios: deque chứa các tỉ lệ detection
        fps: Số frame trên giây (default: 30)
        arrival_threshold: Ngưỡng tỉ lệ để xác định người tới (%)
        departure_threshold: Ngưỡng tỉ lệ để xác định người rời (%)
    
    Returns:
        Dict: Kết quả phân tích với thời gian ước tính
    """
    if not detection_ratios:
        return {
            "mean": 0.0,
            "median": 0.0,
            "mode": 0.0,
            "arrival_detection_time": float('inf'),
            "departure_detection_time": float('inf'),
            "status": "no_data"
        }
    
    stats = calculate_all_statistics(detection_ratios)
    
    # Sử dụng median làm giá trị chính để đánh giá
    median_ratio = stats["median"]
    
    # Ước tính thời gian cần để phát hiện dựa trên độ lệch so với ngưỡng
    if median_ratio > 0:
        # Giả định tốc độ thay đổi tỉ lệ là tuyến tính
        frames_for_arrival = max(0, (arrival_threshold - median_ratio) / (median_ratio / len(detection_ratios)))
        frames_for_departure = max(0, (median_ratio - departure_threshold) / (median_ratio / len(detection_ratios)))
        
        arrival_time_sec = frames_for_arrival / fps if frames_for_arrival > 0 else 0
        departure_time_sec = frames_for_departure / fps if frames_for_departure > 0 else 0
    else:
        arrival_time_sec = float('inf')
        departure_time_sec = float('inf')
    
    # Xác định trạng thái hiện tại dựa trên median
    if median_ratio >= arrival_threshold:
        status = "person_present"
    elif median_ratio <= departure_threshold:
        status = "person_absent"
    else:
        status = "transitioning"
    
    return {
        "mean": round(stats["mean"], 2),
        "median": round(stats["median"], 2),
        "mode": round(stats["mode"], 2),
        "arrival_detection_time_sec": round(arrival_time_sec, 2),
        "departure_detection_time_sec": round(departure_time_sec, 2),
        "current_status": status,
        "data_points": len(detection_ratios)
    }


# ==== MAIN CLASS ====

class HumanPresenceTracker:
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG
        self.ratios: Deque[float] = deque(maxlen=self.config['max_history_size'])
        self.frames_with_person = 0
        self.frames_without_person = 0
        self.person_present = False
        self.last_status_change_time = time.time()

    def update(self, boxes: List[List[float]], frame_shape: Tuple[int, int]) -> Tuple[bool, str, Dict[str, float], Dict[str, Any]]:
        current_time = time.time()
        has_person = len(boxes) > 0

        # --- Update detection history ---
        if has_person:
            self.frames_with_person += 1
            self.frames_without_person = 0
            for box in boxes:
                ratio = calculate_box_area_ratio(box, frame_shape)
                self.ratios.append(ratio)
        else:
            self.frames_without_person += 1
            self.frames_with_person = 0

        # --- Calculate statistics ---
        stats = calculate_all_statistics(self.ratios, self.config['mode_tolerance'])
        
        # --- Calculate time analysis ---
        time_analysis = determine_presence_time(
            self.ratios, 
            self.config['fps'],
            self.config['arrival_median_threshold'],
            self.config['departure_median_threshold']
        )

        # --- Check for ARRIVAL ---
        if (not self.person_present and
            self.frames_with_person >= self.config['min_detection_frames'] and
            stats['median'] > self.config['arrival_median_threshold'] and
            stats['mode'] > self.config['arrival_mode_threshold']):

            if current_time - self.last_status_change_time >= self.config['arrival_time_seconds']:
                self.person_present = True
                self.last_status_change_time = current_time
                return True, "arrival", stats, time_analysis

        # --- Check for DEPARTURE ---
        if (self.person_present and
            self.frames_without_person >= self.config['min_non_detection_frames'] and
            stats['median'] < self.config['departure_median_threshold'] and
            stats['mode'] < self.config['departure_mode_threshold']):

            if current_time - self.last_status_change_time >= self.config['departure_time_seconds']:
                self.person_present = False
                self.last_status_change_time = current_time
                return False, "departure", stats, time_analysis

        # --- No status change ---
        return self.person_present, "none", stats, time_analysis

    def get_state(self) -> Dict:
        """Get current state with comprehensive statistics"""
        stats = calculate_all_statistics(self.ratios, self.config['mode_tolerance'])
        time_analysis = determine_presence_time(
            self.ratios, 
            self.config['fps'],
            self.config['arrival_median_threshold'],
            self.config['departure_median_threshold']
        )
        
        return {
            "person_present": self.person_present,
            "frames_with_person": self.frames_with_person,
            "frames_without_person": self.frames_without_person,
            "history_size": len(self.ratios),
            "statistics": {k: round(v, 2) for k, v in stats.items()},
            "time_analysis": time_analysis,
            "last_status_change_time": self.last_status_change_time
        }

    def detect_person_arrival_departure(
        self,
        boxes: List[List[float]], 
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Phát hiện người tới hoặc rời đi dựa trên phân tích tỉ lệ diện tích
        
        Args:
            boxes: Danh sách các bounding boxes [[x1, y1, x2, y2], ...]
            frame_shape: (height, width) của khung hình
        
        Returns:
            Dict: Kết quả phân tích và quyết định
        """
        # Tính tỉ lệ cho các boxes hiện tại
        current_ratios = []
        for box in boxes:
            ratio = calculate_box_area_ratio(box, frame_shape)
            current_ratios.append(ratio)
            self.ratios.append(ratio)
        
        # Tính tất cả các thống kê
        stats = calculate_all_statistics(self.ratios, self.config['mode_tolerance'])
        
        # Kiểm tra điều kiện arrival (dựa trên median và mode)
        arrival_detected = (
            len(self.ratios) >= self.config['min_detection_frames'] and
            stats["median"] > self.config['arrival_median_threshold'] and
            stats["mode"] > self.config['arrival_mode_threshold']
        )
        
        # Kiểm tra điều kiện departure (dựa trên median và mode)
        departure_detected = (
            len(current_ratios) == 0 and  # Không có detection hiện tại
            len(self.ratios) >= self.config['min_non_detection_frames'] and
            stats["median"] < self.config['departure_median_threshold'] and
            stats["mode"] < self.config['departure_mode_threshold']
        )
        
        # Xác định trạng thái
        if arrival_detected:
            status = "arrival_detected"
        elif departure_detected:
            status = "departure_detected"
        elif len(current_ratios) > 0:
            status = "person_present"
        else:
            status = "no_person"
        
        return {
            "status": status,
            "statistics": {k: round(v, 2) for k, v in stats.items()},
            "current_ratios": [round(r, 2) for r in current_ratios],
            "history_size": len(self.ratios),
            "detection_time_analysis": determine_presence_time(
                self.ratios, self.config['fps'],
                self.config['arrival_median_threshold'],
                self.config['departure_median_threshold']
            )
        }