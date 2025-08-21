#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>

// ==== CONFIGURATION ====
struct Config {
    double arrival_median_threshold = 30.0;
    double arrival_mode_threshold = 20.0;
    double departure_median_threshold = 50.0;
    double departure_mode_threshold = 40.0;
    int min_detection_frames = 20;
    int min_non_detection_frames = 18;
    size_t max_history_size = 1000;
    double arrival_time_seconds = 2.0;
    double departure_time_seconds = 3.0;
    double mode_tolerance = 1.0;
    int fps = 30;
};

// ==== UTILITY FUNCTIONS ====

double calculate_box_area_ratio(const std::vector<double>& box, const std::pair<int, int>& frame_shape) {
    int frame_height = frame_shape.first;
    int frame_width = frame_shape.second;
    double frame_area = frame_width * frame_height;
    
    double box_width = box[2] - box[0];  // x_max - x_min
    double box_height = box[3] - box[1]; // y_max - y_min
    double box_area = box_width * box_height;
    
    return (box_area / frame_area) * 100.0;
}

double calculate_mean_ratio(const std::deque<double>& detection_ratios) {
    if (detection_ratios.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(detection_ratios.begin(), detection_ratios.end(), 0.0);
    return sum / detection_ratios.size();
}

double calculate_median_ratio(std::deque<double> detection_ratios) {
    if (detection_ratios.empty()) {
        return 0.0;
    }
    
    std::sort(detection_ratios.begin(), detection_ratios.end());
    size_t n = detection_ratios.size();
    
    if (n % 2 == 1) {
        return detection_ratios[n / 2];
    } else {
        size_t mid = n / 2;
        return (detection_ratios[mid - 1] + detection_ratios[mid]) / 2.0;
    }
}

double calculate_mode_ratio(const std::deque<double>& detection_ratios, double tolerance = 1.0) {
    if (detection_ratios.empty()) {
        return 0.0;
    }
    
    // Round values and count frequencies
    std::map<double, int> frequency_map;
    for (double ratio : detection_ratios) {
        double rounded_ratio = std::round(ratio / tolerance) * tolerance;
        frequency_map[rounded_ratio]++;
    }
    
    // Find the mode
    double mode = 0.0;
    int max_count = 0;
    bool has_unique_mode = true;
    
    for (const auto& pair : frequency_map) {
        if (pair.second > max_count) {
            mode = pair.first;
            max_count = pair.second;
            has_unique_mode = true;
        } else if (pair.second == max_count) {
            has_unique_mode = false;
        }
    }
    
    if (has_unique_mode && max_count > 1) {
        return mode;
    } else {
        // Return median if no unique mode
        return calculate_median_ratio(detection_ratios);
    }
}

struct Statistics {
    double mean;
    double median;
    double mode;
};

Statistics calculate_all_statistics(const std::deque<double>& detection_ratios, double tolerance = 1.0) {
    return {
        calculate_mean_ratio(detection_ratios),
        calculate_median_ratio(detection_ratios),
        calculate_mode_ratio(detection_ratios, tolerance)
    };
}

struct TimeAnalysis {
    double mean;
    double median;
    double mode;
    double arrival_detection_time_sec;
    double departure_detection_time_sec;
    std::string current_status;
    int data_points;
};

TimeAnalysis determine_presence_time(
    const std::deque<double>& detection_ratios,
    int fps = 30,
    double arrival_threshold = 30.0,
    double departure_threshold = 50.0
) {
    if (detection_ratios.empty()) {
        return {
            0.0, 0.0, 0.0,
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            "no_data",
            0
        };
    }
    
    Statistics stats = calculate_all_statistics(detection_ratios);
    double median_ratio = stats.median;
    
    double arrival_time_sec, departure_time_sec;
    std::string status;
    
    if (median_ratio > 0) {
        double change_rate = median_ratio / detection_ratios.size();
        double frames_for_arrival = std::max(0.0, (arrival_threshold - median_ratio) / change_rate);
        double frames_for_departure = std::max(0.0, (median_ratio - departure_threshold) / change_rate);
        
        arrival_time_sec = (frames_for_arrival > 0) ? frames_for_arrival / fps : 0;
        departure_time_sec = (frames_for_departure > 0) ? frames_for_departure / fps : 0;
    } else {
        arrival_time_sec = std::numeric_limits<double>::infinity();
        departure_time_sec = std::numeric_limits<double>::infinity();
    }
    
    if (median_ratio >= arrival_threshold) {
        status = "person_present";
    } else if (median_ratio <= departure_threshold) {
        status = "person_absent";
    } else {
        status = "transitioning";
    }
    
    return {
        std::round(stats.mean * 100) / 100,
        std::round(stats.median * 100) / 100,
        std::round(stats.mode * 100) / 100,
        std::round(arrival_time_sec * 100) / 100,
        std::round(departure_time_sec * 100) / 100,
        status,
        static_cast<int>(detection_ratios.size())
    };
}

// ==== MAIN CLASS ====

class HumanPresenceTracker {
private:
    Config config;
    std::deque<double> ratios;
    int frames_with_person;
    int frames_without_person;
    bool person_present;
    std::chrono::steady_clock::time_point last_status_change_time;

    void maintain_history_size() {
        while (ratios.size() > config.max_history_size) {
            ratios.pop_front();
        }
    }

public:
    HumanPresenceTracker(const Config& cfg = Config{}) 
        : config(cfg), 
          frames_with_person(0), 
          frames_without_person(0), 
          person_present(false),
          last_status_change_time(std::chrono::steady_clock::now()) {
    }

    struct UpdateResult {
        bool person_present;
        std::string status;
        Statistics stats;
        TimeAnalysis time_analysis;
    };

    UpdateResult update(const std::vector<std::vector<double>>& boxes, const std::pair<int, int>& frame_shape) {
        auto current_time = std::chrono::steady_clock::now();
        bool has_person = !boxes.empty();

        // Update detection history
        if (has_person) {
            frames_with_person++;
            frames_without_person = 0;
            for (const auto& box : boxes) {
                double ratio = calculate_box_area_ratio(box, frame_shape);
                ratios.push_back(ratio);
                maintain_history_size();
            }
        } else {
            frames_without_person++;
            frames_with_person = 0;
        }

        // Calculate statistics
        Statistics stats = calculate_all_statistics(ratios, config.mode_tolerance);
        
        // Calculate time analysis
        TimeAnalysis time_analysis = determine_presence_time(
            ratios, 
            config.fps,
            config.arrival_median_threshold,
            config.departure_median_threshold
        );

        double elapsed_seconds = std::chrono::duration<double>(
            current_time - last_status_change_time).count();

        // Check for ARRIVAL
        if (!person_present &&
            frames_with_person >= config.min_detection_frames &&
            stats.median > config.arrival_median_threshold &&
            stats.mode > config.arrival_mode_threshold &&
            elapsed_seconds >= config.arrival_time_seconds) {
            
            person_present = true;
            last_status_change_time = current_time;
            return {true, "arrival", stats, time_analysis};
        }

        // Check for DEPARTURE
        if (person_present &&
            frames_without_person >= config.min_non_detection_frames &&
            stats.median < config.departure_median_threshold &&
            stats.mode < config.departure_mode_threshold &&
            elapsed_seconds >= config.departure_time_seconds) {
            
            person_present = false;
            last_status_change_time = current_time;
            return {false, "departure", stats, time_analysis};
        }

        // No status change
        return {person_present, "none", stats, time_analysis};
    }

    struct State {
        bool person_present;
        int frames_with_person;
        int frames_without_person;
        size_t history_size;
        Statistics statistics;
        TimeAnalysis time_analysis;
        std::chrono::steady_clock::time_point last_status_change_time;
    };

    State get_state() {
        Statistics stats = calculate_all_statistics(ratios, config.mode_tolerance);
        TimeAnalysis time_analysis = determine_presence_time(
            ratios, 
            config.fps,
            config.arrival_median_threshold,
            config.departure_median_threshold
        );
        
        return {
            person_present,
            frames_with_person,
            frames_without_person,
            ratios.size(),
            stats,
            time_analysis,
            last_status_change_time
        };
    }

    struct DetectionResult {
        std::string status;
        Statistics statistics;
        std::vector<double> current_ratios;
        size_t history_size;
        TimeAnalysis detection_time_analysis;
    };

    DetectionResult detect_person_arrival_departure(
        const std::vector<std::vector<double>>& boxes, 
        const std::pair<int, int>& frame_shape
    ) {
        // Calculate ratios for current boxes
        std::vector<double> current_ratios;
        for (const auto& box : boxes) {
            double ratio = calculate_box_area_ratio(box, frame_shape);
            current_ratios.push_back(ratio);
            ratios.push_back(ratio);
            maintain_history_size();
        }

        // Calculate all statistics
        Statistics stats = calculate_all_statistics(ratios, config.mode_tolerance);
        
        // Check arrival conditions
        bool arrival_detected = (
            ratios.size() >= config.min_detection_frames &&
            stats.median > config.arrival_median_threshold &&
            stats.mode > config.arrival_mode_threshold
        );
        
        // Check departure conditions
        bool departure_detected = (
            current_ratios.empty() &&
            ratios.size() >= config.min_non_detection_frames &&
            stats.median < config.departure_median_threshold &&
            stats.mode < config.departure_mode_threshold
        );
        
        // Determine status
        std::string status;
        if (arrival_detected) {
            status = "arrival_detected";
        } else if (departure_detected) {
            status = "departure_detected";
        } else if (!current_ratios.empty()) {
            status = "person_present";
        } else {
            status = "no_person";
        }
        
        return {
            status,
            stats,
            current_ratios,
            ratios.size(),
            determine_presence_time(
                ratios, config.fps,
                config.arrival_median_threshold,
                config.departure_median_threshold
            )
        };
    }
};

// Helper function to print statistics
void print_statistics(const Statistics& stats) {
    std::cout << "Mean: " << stats.mean 
              << ", Median: " << stats.median 
              << ", Mode: " << stats.mode << std::endl;
}

int main() {
    // Example usage
    Config config;
    HumanPresenceTracker tracker(config);
    
    std::vector<std::vector<double>> boxes = {{10, 10, 50, 50}};
    std::pair<int, int> frame_shape = {480, 640};
    
    auto result = tracker.update(boxes, frame_shape);
    std::cout << "Status: " << result.status << std::endl;
    print_statistics(result.stats);
    
    return 0;
}