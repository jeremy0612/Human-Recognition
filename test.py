import ast
from typing import List, Dict, Any, Tuple
from collections import deque
import statistics
import time

# ==== CÁC HÀM TIỆN ÍCH ====
def calculate_box_area_ratio(box: List[float], frame_width: int, frame_height: int) -> float:
    """
    Tính tỉ lệ diện tích của box so với toàn bộ khung hình
    """
    frame_area = frame_width * frame_height
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    box_area = box_width * box_height
    
    return (box_area / frame_area) * 100.0

def calculate_statistics(ratios: deque, tolerance: float = 1.0) -> Dict[str, float]:
    """
    Tính các thống kê từ danh sách tỉ lệ
    """
    if not ratios:
        return {"mean": 0.0, "median": 0.0, "mode": 0.0}
    
    ratios_list = list(ratios)
    
    # Tính mean
    mean_val = statistics.mean(ratios_list) if ratios_list else 0.0
    
    # Tính median
    median_val = statistics.median(ratios_list) if ratios_list else 0.0
    
    # Tính mode với tolerance
    try:
        rounded_ratios = [round(ratio / tolerance) * tolerance for ratio in ratios_list]
        mode_val = statistics.mode(rounded_ratios)
    except statistics.StatisticsError:
        mode_val = statistics.median(rounded_ratios) if rounded_ratios else 0.0
    
    return {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "mode": round(mode_val, 2)
    }

# ==== CLASS TRACKER ====
class PersonDetectionTracker:
    """
    Class theo dõi phát hiện người và xác định thời điểm đến/rời
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detection_ratios = deque(maxlen=1000)
        self.frames_without_person = 0
        self.person_detected = False
        self.last_detection_time = 0
        self.arrival_time_threshold = self.config.get('arrival_time_seconds', 2.0)
        self.departure_time_threshold = self.config.get('departure_time_seconds', 3.0)
    
    def update_detection(self, boxes: List[List[float]], frame_width: int, frame_height: int) -> Tuple[bool, str, Dict[str, float]]:
        """
        Cập nhật trạng thái detection và xác định hành động
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
        
        # Tính thống kê
        stats = calculate_statistics(self.detection_ratios)
        
        # Lấy các ngưỡng từ config
        arrival_threshold = self.config.get('welcome_threshold', 20)
        departure_threshold = self.config.get('departure_threshold', 18)
        median_arrival_threshold = self.config.get('median_ratio_threshold', 30.0)
        mode_arrival_threshold = self.config.get('mode_ratio_threshold', 20.0)
        median_departure_threshold = self.config.get('departure_median_threshold', 50.0)
        mode_departure_threshold = self.config.get('departure_mode_threshold', 40.0)
        
        # Xác định hành động
        action = "none"
        
        # Kiểm tra điều kiện khách đến
        if (len(self.detection_ratios) >= arrival_threshold and 
            not self.person_detected and 
            stats['median'] > median_arrival_threshold and 
            stats['mode'] > mode_arrival_threshold):
            action = "arrival"
        
        # Kiểm tra điều kiện khách rời đi
        elif (self.frames_without_person >= departure_threshold and 
              self.person_detected and 
              stats['median'] < median_departure_threshold and 
              stats['mode'] < mode_departure_threshold):
            action = "departure"
        
        # Kiểm tra thời gian để xác nhận thay đổi trạng thái
        if action == "arrival":
            if current_time - self.last_detection_time >= self.arrival_time_threshold:
                self.person_detected = True
                self.last_detection_time = current_time
                return True, "arrival", stats
        
        elif action == "departure":
            if current_time - self.last_detection_time >= self.departure_time_threshold:
                self.person_detected = False
                self.last_detection_time = current_time
                return False, "departure", stats
        
        return self.person_detected, "none", stats
    
    def get_state(self) -> Dict[str, Any]:
        """Lấy trạng thái hiện tại"""
        stats = calculate_statistics(self.detection_ratios)
        return {
            "person_detected": self.person_detected,
            "total_detections": len(self.detection_ratios),
            "frames_without_person": self.frames_without_person,
            "statistics": stats,
            "last_detection_time": self.last_detection_time
        }

# ==== HÀM PARSE VÀ TEST ====
def parse_detection_line(line: str, frame_width: int = 640, frame_height: int = 480) -> List[List[float]]:
    """
    Parse dòng detection từ định dạng CSV thành boxes
    Format: 1,-1,772.68,455.43,41.871,127.61,2.1262,-1,-1,-1
    Giả sử: track_id, class_id, x_center, y_center, width, height, confidence, ...
    """
    try:
        parts = line.strip().split(',')
        if len(parts) < 7:
            return []
        
        # Lấy thông tin box
        x_center = float(parts[2])
        y_center = float(parts[3])
        width = float(parts[4])
        height = float(parts[5])
        confidence = float(parts[6])
        
        # Chỉ xử lý nếu confidence > ngưỡng
        if confidence < 0.5:
            return []
        
        # Chuyển từ center coordinates sang bounding box coordinates
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        # Đảm bảo box nằm trong frame
        x_min = max(0, min(x_min, frame_width))
        y_min = max(0, min(y_min, frame_height))
        x_max = max(0, min(x_max, frame_width))
        y_max = max(0, min(y_max, frame_height))
        
        return [[x_min, y_min, x_max, y_max]]
        
    except (ValueError, IndexError) as e:
        print(f"Lỗi parse detection line: {e}")
        return []

def test_person_detection_from_txt(file_path: str, config: Dict = None) -> List[Dict[str, Any]]:
    """
    Test PersonDetectionTracker từ file txt với định dạng detection
    """
    # Khởi tạo tracker
    tracker = PersonDetectionTracker(config)
    
    results = []
    frame_width, frame_height = 640, 480  # Kích thước frame mặc định
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Parse dòng detection
                    boxes = parse_detection_line(line, frame_width, frame_height)
                    
                    # Cập nhật tracker
                    person_detected, status, stats = tracker.update_detection(boxes, frame_width, frame_height)
                    
                    # Lưu kết quả
                    result = {
                        'frame': i + 1,
                        'input_line': line,
                        'boxes': boxes,
                        'frame_size': (frame_width, frame_height),
                        'person_detected': person_detected,
                        'status': status,
                        'statistics': stats,
                        'state': tracker.get_state()
                    }
                    
                    results.append(result)
                    
                    # In thông tin để debug
                    print(f"Frame {i+1}: {status.upper()}")
                    print(f"  Input: {line}")
                    print(f"  Boxes: {len(boxes)}, Confidence: {float(line.split(',')[6]) if len(line.split(',')) > 6 else 'N/A'}")
                    print(f"  Person detected: {person_detected}")
                    print(f"  Stats: mean={stats['mean']:.2f}%, median={stats['median']:.2f}%, mode={stats['mode']:.2f}%")
                    if boxes:
                        area_ratio = calculate_box_area_ratio(boxes[0], frame_width, frame_height)
                        print(f"  Area ratio: {area_ratio:.2f}%")
                    print("-" * 60)
                    
                except (ValueError, TypeError) as e:
                    print(f"Lỗi parse dòng {i+1}: {e}")
                    print(f"Dòng lỗi: {line}")
                    continue
                    
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        return []
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return []
    
    return results

def analyze_detection_file(file_path: str):
    """
    Phân tích file detection để hiểu định dạng
    """
    print("Phân tích file detection...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            print(f"Tổng số dòng: {len(lines)}")
            
            # Phân tích 5 dòng đầu
            for i, line in enumerate(lines[:5]):
                line = line.strip()
                if line:
                    parts = line.split(',')
                    print(f"Dòng {i+1}: {len(parts)} cột")
                    
                    # Thử parse numbers
                    try:
                        numbers = [float(p) for p in parts if p.strip()]
                        print(f"  Values: {numbers}")
                        if len(numbers) >= 7:
                            print(f"  Confidence: {numbers[6]}")
                    except:
                        print("  Không thể convert sang số")
    
    except Exception as e:
        print(f"Lỗi phân tích file: {e}")

def generate_test_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tạo báo cáo tổng hợp từ kết quả test
    """
    if not results:
        return {"error": "No results to analyze"}
    
    total_frames = len(results)
    arrival_count = sum(1 for r in results if r['status'] == 'arrival')
    departure_count = sum(1 for r in results if r['status'] == 'departure')
    frames_with_detection = sum(1 for r in results if r['boxes'])
    
    return {
        "total_frames": total_frames,
        "frames_with_detection": frames_with_detection,
        "frames_without_detection": total_frames - frames_with_detection,
        "arrival_events": arrival_count,
        "departure_events": departure_count,
        "final_person_detected": results[-1]['person_detected'] if results else False
    }

def save_results_to_file(results: List[Dict[str, Any]], output_file: str):
    """Lưu kết quả ra file txt"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("KẾT QUẢ TEST PERSON DETECTION TRACKER\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"Frame {result['frame']}:\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Person detected: {result['person_detected']}\n")
            f.write(f"  Input: {result['input_line']}\n")
            f.write(f"  Boxes: {len(result['boxes'])}\n")
            f.write(f"  Statistics: mean={result['statistics']['mean']:.2f}%, "
                   f"median={result['statistics']['median']:.2f}%, "
                   f"mode={result['statistics']['mode']:.2f}%\n")
            if result['boxes']:
                area_ratio = calculate_box_area_ratio(result['boxes'][0], result['frame_size'][0], result['frame_size'][1])
                f.write(f"  Area ratio: {area_ratio:.2f}%\n")
            f.write("-" * 40 + "\n")
        
        # Thêm báo cáo tổng hợp
        report = generate_test_report(results)
        f.write("\nBÁO CÁO TỔNG HỢP:\n")
        f.write("=" * 40 + "\n")
        for key, value in report.items():
            f.write(f"{key}: {value}\n")

# ==== MAIN ====
if __name__ == "__main__":
    # Test với file txt
    test_file = "det.txt"
    output_file = "detection_results.txt"
    
    print("Bắt đầu test Person Detection Tracker...")
    print(f"Đọc file: {test_file}")
    
    # Phân tích file trước
    analyze_detection_file(test_file)
    print("\n" + "="*60)
    
    # Cấu hình tracker
    config = {
        'welcome_threshold': 5,
        'departure_threshold': 10,
        'median_ratio_threshold': 25.0,
        'mode_ratio_threshold': 20.0,
        'departure_median_threshold': 15.0,
        'departure_mode_threshold': 10.0,
        'arrival_time_seconds': 1.0,
        'departure_time_seconds': 2.0
    }
    
    # Chạy test
    results = test_person_detection_from_txt(test_file, config)
    
    # Lưu kết quả
    if results:
        save_results_to_file(results, output_file)
        print(f"\nĐã lưu kết quả vào: {output_file}")
        
        # Tạo báo cáo
        report = generate_test_report(results)
        print("\n" + "="*60)
        print("BÁO CÁO TỔNG HỢP TEST")
        print("="*60)
        print(f"Tổng số frame: {report['total_frames']}")
        print(f"Frame có detection: {report['frames_with_detection']}")
        print(f"Frame không detection: {report['frames_without_detection']}")
        print(f"Số lần arrival: {report['arrival_events']}")
        print(f"Số lần departure: {report['departure_events']}")
        print(f"Trạng thái cuối: {'CÓ NGƯỜI' if report['final_person_detected'] else 'KHÔNG CÓ NGƯỜI'}")
    else:
        print("Không có kết quả để phân tích")