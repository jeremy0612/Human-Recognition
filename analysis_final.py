import statistics
from collections import deque
from typing import List, Tuple, Dict, Any
import time

def calculate_box_area_ratio(box: List[float], frame_width: int, frame_height: int) -> float:
    """
    Tính tỉ lệ diện tích của box so với toàn bộ khung hình
    
    Args:
        box: [x_min, y_min, x_max, y_max] - tọa độ bounding box
        frame_width: chiều rộng khung hình
        frame_height: chiều cao khung hình
    
    Returns:
        Tỉ lệ diện tích tính bằng phần trăm
    """
    frame_area = frame_width * frame_height
    box_width = box[2] - box[0]  # x_max - x_min
    box_height = box[3] - box[1]  # y_max - y_min
    box_area = box_width * box_height
    
    return (box_area / frame_area) * 100.0


def calculate_statistics(ratios: deque, tolerance: float = 1.0) -> Dict[str, float]:
    """
    Tính các thống kê từ danh sách tỉ lệ
    
    Args:
        ratios: deque chứa các tỉ lệ diện tích
        tolerance: dung sai để làm tròn khi tính mode
    
    Returns:
        Dictionary chứa mean, median, mode
    """
    if not ratios:
        return {"mean": 0.0, "median": 0.0, "mode": 0.0}
    
    # Chuyển deque sang list
    ratios_list = list(ratios)
    
    # Tính mean (trung bình)
    mean_val = statistics.mean(ratios_list) if ratios_list else 0.0
    
    # Tính median (trung vị)
    median_val = statistics.median(ratios_list) if ratios_list else 0.0
    
    # Tính mode (giá trị xuất hiện nhiều nhất) với tolerance
    try:
        rounded_ratios = [round(ratio / tolerance) * tolerance for ratio in ratios_list]
        mode_val = statistics.mode(rounded_ratios)
    except statistics.StatisticsError:
        # Nếu không có mode duy nhất, sử dụng median
        mode_val = statistics.median(rounded_ratios) if rounded_ratios else 0.0
    
    return {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "mode": round(mode_val, 2)
    }


def determine_person_status(ratios: deque, config: Dict[str, Any], 
                           current_status: bool, frames_without_person: int) -> Tuple[bool, str]:
    """
    Xác định trạng thái người (đến hay rời đi) dựa trên thống kê
    
    Args:
        ratios: deque chứa các tỉ lệ diện tích
        config: dictionary cấu hình với các ngưỡng
        current_status: trạng thái hiện tại (True = có người)
        frames_without_person: số frame không có người liên tiếp
    
    Returns:
        Tuple (new_status, action)
        - new_status: trạng thái mới
        - action: "arrival", "departure", hoặc "none"
    """
    stats = calculate_statistics(ratios)
    
    # Lấy các ngưỡng từ config, với giá trị mặc định
    arrival_threshold = config.get('welcome_threshold', 20)
    departure_threshold = config.get('departure_threshold', 18)
    median_arrival_threshold = config.get('median_ratio_threshold', 30.0)
    mode_arrival_threshold = config.get('mode_ratio_threshold', 20.0)
    median_departure_threshold = config.get('departure_median_threshold', 50.0)
    mode_departure_threshold = config.get('departure_mode_threshold', 40.0)
    
    # Kiểm tra điều kiện khách đến
    if (len(ratios) >= arrival_threshold and 
        not current_status and 
        stats['median'] > median_arrival_threshold and 
        stats['mode'] > mode_arrival_threshold):
        return True, "arrival"
    
    # Kiểm tra điều kiện khách rời đi
    elif (frames_without_person > departure_threshold and 
          current_status and 
          stats['median'] < median_departure_threshold and 
          stats['mode'] < mode_departure_threshold):
        return False, "departure"
    
    # Không có thay đổi
    return current_status, "none"


class PersonDetectionTracker:
    """
    Class theo dõi phát hiện người và xác định thời điểm đến/rời
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detection_ratios = deque(maxlen=1000)  # Lưu 1000 tỉ lệ gần nhất
        self.frames_without_person = 0
        self.person_detected = False
        self.last_detection_time = 0
        self.arrival_time_threshold = self.config.get('arrival_time_seconds', 2.0)
        self.departure_time_threshold = self.config.get('departure_time_seconds', 3.0)
    
    def update_detection(self, boxes: List[List[float]], frame_width: int, frame_height: int) -> Tuple[bool, str]:
        """
        Cập nhật trạng thái detection và xác định hành động
        
        Args:
            boxes: danh sách các bounding boxes [x_min, y_min, x_max, y_max]
            frame_width: chiều rộng frame
            frame_height: chiều cao frame
        
        Returns:
            Tuple (person_detected, action)
        """
        current_time = time.time()
        has_person = len(boxes) > 0
        
        # Cập nhật tỉ lệ nếu có người
        if has_person:
            self.frames_without_person = 0
            for box in boxes:
                ratio = calculate_box_area_ratio(box, frame_width, frame_height)
                self.detection_ratios.append(ratio)
        else:
            self.frames_without_person += 1
        
        # Xác định trạng thái mới
        new_status, action = determine_person_status(
            self.detection_ratios, self.config, self.person_detected, self.frames_without_person
        )
        
        # Kiểm tra thời gian để xác nhận thay đổi trạng thái
        if action == "arrival":
            # Kiểm tra nếu đủ thời gian để xác nhận khách đến
            if current_time - self.last_detection_time >= self.arrival_time_threshold:
                self.person_detected = True
                self.last_detection_time = current_time
                return True, "arrival"
        
        elif action == "departure":
            # Kiểm tra nếu đủ thời gian để xác nhận khách rời đi
            if current_time - self.last_detection_time >= self.departure_time_threshold:
                self.person_detected = False
                self.last_detection_time = current_time
                return False, "departure"
        
        return self.person_detected, "none"
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Lấy thống kê hiện tại"""
        stats = calculate_statistics(self.detection_ratios)
        stats.update({
            "total_detections": len(self.detection_ratios),
            "frames_without_person": self.frames_without_person,
            "person_detected": self.person_detected
        })
        return stats
