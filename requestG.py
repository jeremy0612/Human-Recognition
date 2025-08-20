import statistics
from typing import List, Tuple, Optional
from collections import deque
from datetime import datetime

# --- Hàm tính tỷ lệ diện tích box so với frame ---
def calculate_area_ratio(box: List[float], frame_shape: Tuple[int, int, int]) -> float:
    frame_area = frame_shape[0] * frame_shape[1]  # height * width
    box_area = (box[2] - box[0]) * (box[3] - box[1])  # (xmax - xmin) * (ymax - ymin)
    return (box_area / frame_area) * 100.0  # trả về phần trăm

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
