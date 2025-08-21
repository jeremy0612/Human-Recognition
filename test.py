import ast
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import statistics
import time
from datetime import datetime

# --- Hàm tính tỷ lệ diện tích box so với frame ---
def calculate_area_ratio(box: List[float], frame_shape: Tuple[int, int]) -> float:
    frame_area = frame_shape[0] * frame_shape[1]  # height * width
    box_area = (box[2] - box[0]) * (box[3] - box[1])  # (xmax - xmin) * (ymax - ymin)
    return (box_area / frame_area) * 100.0  # trả về phần trăm

# Alias function for compatibility
def calculate_box_area_ratio(box: List[float], frame_width: int, frame_height: int) -> float:
    return calculate_area_ratio(box, (frame_height, frame_width))

# --- Hàm tính thống kê ---
def calculate_statistics(ratios: List[float], mode_tolerance: float = 1.0) -> Tuple[float, float, float]:
    if not ratios:
        return 0.0, 0.0, 0.0

    mean_ratio = sum(ratios) / len(ratios)
    median_ratio = statistics.median(ratios)

    # Làm tròn theo tolerance để tìm mode
    rounded_ratios = [round(r / mode_tolerance) * mode_tolerance for r in ratios]
    try:
        mode_ratio = statistics.mode(rounded_ratios)
    except statistics.StatisticsError:
        mode_ratio = median_ratio  # fallback nếu không có mode rõ ràng

    return mean_ratio, median_ratio, mode_ratio

class PersonDetectionTracker:
    def __init__(self, config: Dict[str, Any] = None):
        # Cấu hình mặc định
        self.config = {
            'welcome_threshold': 5,
            'departure_threshold': 10,
            'median_ratio_threshold': 25.0,
            'mode_ratio_threshold': 20.0,
            'departure_median_threshold': 15.0,
            'departure_mode_threshold': 10.0,
            'arrival_time_seconds': 1.0,
            'departure_time_seconds': 2.0,
            'history_size': 30
        }
        
        # Cập nhật cấu hình nếu có
        if config:
            self.config.update(config)
            
        # Trạng thái
        self.person_detected = False
        self.detection_history = deque(maxlen=self.config['history_size'])
        self.last_detection_time = None
        self.human_count = 0
        self.non_human_count = 0
        
    def update_detection(self, boxes: List[List[float]], frame_width: int, frame_height: int) -> Tuple[bool, str, Dict[str, float]]:
        """
        Cập nhật detection và trả về:
        - person_detected: True nếu có người được phát hiện
        - status: 'arrival', 'departure', hoặc 'no_change'
        - statistics: các thống kê về tỷ lệ diện tích
        """
        frame_time = datetime.now()
        frame_shape = (frame_height, frame_width)
        
        # Tính tỷ lệ diện tích cho các boxes
        ratios = [calculate_area_ratio(box, frame_shape) for box in boxes]
        
        # Cập nhật lịch sử
        if ratios:
            self.detection_history.extend(ratios)
        
        # Tính toán thống kê
        mean_ratio, median_ratio, mode_ratio = calculate_statistics(list(self.detection_history))
        stats = {
            'mean': mean_ratio,
            'median': median_ratio,
            'mode': mode_ratio
        }
        
        # Xác định trạng thái hiện tại
        current_human_detected = len(boxes) > 0
        
        # Cập nhật bộ đếm
        if current_human_detected:
            self.human_count += 1
            self.non_human_count = 0
        else:
            self.non_human_count += 1
            self.human_count = 0
        
        # Kiểm tra sự kiện đến/rời đi
        event = detect_arrival_or_departure(
            self.detection_history,
            frame_time,
            self.human_count,
            self.non_human_count,
            self.person_detected,
            self.config
        )
        
        # Cập nhật trạng thái
        if event == 'arrival':
            self.person_detected = True
            status = 'arrival'
        elif event == 'departure':
            self.person_detected = False
            status = 'departure'
        else:
            status = 'no_change'
        
        return self.person_detected, status, stats
    
    def get_state(self) -> Dict[str, Any]:
        """Trả về trạng thái hiện tại của tracker"""
        return {
            'person_detected': self.person_detected,
            'history_size': len(self.detection_history),
            'human_count': self.human_count,
            'non_human_count': self.non_human_count
        }

# --- Hàm xác định người đến hoặc rời ---
def detect_arrival_or_departure(
    detection_ratios: deque,
    frame_time: datetime,
    human_count: int,
    non_human_count: int,
    current_human_detected: bool,
    config: dict
) -> Optional[str]:
    """
    Trả về: 'arrival', 'departure', hoặc None
    """

    mean_ratio, median_ratio, mode_ratio = calculate_statistics(list(detection_ratios))

    # Điều kiện người đến
    if (human_count >= config['welcome_threshold'] and
        not current_human_detected and
        median_ratio > config['median_ratio_threshold'] and
        mode_ratio > config['mode_ratio_threshold']):
        print(f"[{frame_time}] 👤 Người đến - Median: {median_ratio:.2f}% - Mode: {mode_ratio:.2f}%")
        return 'arrival'

    # Điều kiện người rời
    elif (non_human_count >= config['departure_threshold'] and
          current_human_detected and
          median_ratio < config['departure_median_threshold'] and
          mode_ratio < config['departure_mode_threshold']):
        print(f"[{frame_time}] 👋 Người rời - Median: {median_ratio:.2f}% - Mode: {mode_ratio:.2f}%")
        return 'departure'

    return None


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