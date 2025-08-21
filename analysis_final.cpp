#include <vector>
#include <deque>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <variant>

using namespace std;
using namespace std::chrono;

// Type aliases for better readability
using Box = vector<float>;
using Config = map<string, float>;
using Stats = map<string, float>;

// Variant type for different stat values
using StatValue = variant<float, int, bool>;

class PersonDetectionTracker {
private:
    deque<float> detection_ratios;
    int frames_without_person;
    bool person_detected;
    system_clock::time_point last_detection_time;
    Config config;
    
    // Maximum size for the deque
    static const size_t MAX_RATIOS = 1000;
    
    // Helper function to calculate statistics
    Stats calculate_statistics(const deque<float>& ratios, float tolerance = 1.0f) {
        Stats stats;
        
        if (ratios.empty()) {
            stats["mean"] = 0.0f;
            stats["median"] = 0.0f;
            stats["mode"] = 0.0f;
            return stats;
        }
        
        vector<float> ratios_vec(ratios.begin(), ratios.end());
        
        // Calculate mean
        float sum = accumulate(ratios_vec.begin(), ratios_vec.end(), 0.0f);
        stats["mean"] = roundf((sum / ratios_vec.size()) * 100) / 100;
        
        // Calculate median
        vector<float> sorted_ratios = ratios_vec;
        sort(sorted_ratios.begin(), sorted_ratios.end());
        
        size_t size = sorted_ratios.size();
        if (size % 2 == 0) {
            stats["median"] = roundf(((sorted_ratios[size/2 - 1] + sorted_ratios[size/2]) / 2.0f) * 100) / 100;
        } else {
            stats["median"] = roundf(sorted_ratios[size/2] * 100) / 100;
        }
        
        // Calculate mode with tolerance
        try {
            unordered_map<int, int> frequency_map;
            for (float ratio : ratios_vec) {
                int key = static_cast<int>(roundf(ratio / tolerance));
                frequency_map[key]++;
            }
            
            int max_count = 0;
            float mode_val = 0.0f;
            for (const auto& pair : frequency_map) {
                if (pair.second > max_count) {
                    max_count = pair.second;
                    mode_val = static_cast<float>(pair.first) * tolerance;
                }
            }
            stats["mode"] = roundf(mode_val * 100) / 100;
        } catch (const exception& e) {
            // Fall back to median if mode calculation fails
            stats["mode"] = stats["median"];
        }
        
        return stats;
    }
    
    // Helper function to get config value with default
    float get_config_value(const string& key, float default_value) {
        auto it = config.find(key);
        if (it != config.end()) {
            return it->second;
        }
        return default_value;
    }

public:
    PersonDetectionTracker(const Config& cfg = Config()) 
        : frames_without_person(0), person_detected(false), 
          last_detection_time(system_clock::now()), config(cfg) {
        detection_ratios = deque<float>();
    }
    
    // Calculate box area ratio
    static float calculate_box_area_ratio(const Box& box, int frame_width, int frame_height) {
        float frame_area = static_cast<float>(frame_width * frame_height);
        float box_width = box[2] - box[0];  // x_max - x_min
        float box_height = box[3] - box[1]; // y_max - y_min
        float box_area = box_width * box_height;
        
        return (box_area / frame_area) * 100.0f;
    }
    
    // Determine person status
    pair<bool, string> determine_person_status(const deque<float>& ratios, 
                                              bool current_status, 
                                              int frames_without_person_count) {
        Stats stats = calculate_statistics(ratios);
        
        // Get thresholds from config with defaults
        float arrival_threshold = get_config_value("welcome_threshold", 20.0f);
        float departure_threshold = get_config_value("departure_threshold", 18.0f);
        float median_arrival_threshold = get_config_value("median_ratio_threshold", 30.0f);
        float mode_arrival_threshold = get_config_value("mode_ratio_threshold", 20.0f);
        float median_departure_threshold = get_config_value("departure_median_threshold", 50.0f);
        float mode_departure_threshold = get_config_value("departure_mode_threshold", 40.0f);
        
        // Check arrival conditions
        if (ratios.size() >= static_cast<size_t>(arrival_threshold) && 
            !current_status && 
            stats.at("median") > median_arrival_threshold && 
            stats.at("mode") > mode_arrival_threshold) {
            return make_pair(true, "arrival");
        }
        
        // Check departure conditions
        else if (frames_without_person_count > static_cast<int>(departure_threshold) && 
                 current_status && 
                 stats.at("median") < median_departure_threshold && 
                 stats.at("mode") < mode_departure_threshold) {
            return make_pair(false, "departure");
        }
        
        // No change
        return make_pair(current_status, "none");
    }
    
    // Update detection
    pair<bool, string> update_detection(const vector<Box>& boxes, int frame_width, int frame_height) {
        auto current_time = system_clock::now();
        bool has_person = !boxes.empty();
        
        // Update ratios if person is detected
        if (has_person) {
            frames_without_person = 0;
            for (const auto& box : boxes) {
                if (box.size() >= 4) { // Ensure box has correct format
                    float ratio = calculate_box_area_ratio(box, frame_width, frame_height);
                    detection_ratios.push_back(ratio);
                    
                    // Maintain maximum size
                    if (detection_ratios.size() > MAX_RATIOS) {
                        detection_ratios.pop_front();
                    }
                }
            }
        } else {
            frames_without_person++;
        }
        
        // Determine new status
        auto status_result = determine_person_status(
            detection_ratios, person_detected, frames_without_person
        );
        bool new_status = status_result.first;
        string action = status_result.second;
        
        // Get time thresholds
        float arrival_time_threshold = get_config_value("arrival_time_seconds", 2.0f);
        float departure_time_threshold = get_config_value("departure_time_seconds", 3.0f);
        
        // Calculate time difference in seconds
        auto time_diff = duration_cast<milliseconds>(
            current_time - last_detection_time
        ).count() / 1000.0f;
        
        // Check time conditions for state changes
        if (action == "arrival") {
            if (time_diff >= arrival_time_threshold) {
                person_detected = true;
                last_detection_time = current_time;
                return make_pair(true, "arrival");
            }
        } else if (action == "departure") {
            if (time_diff >= departure_time_threshold) {
                person_detected = false;
                last_detection_time = current_time;
                return make_pair(false, "departure");
            }
        }
        
        return make_pair(person_detected, "none");
    }
    
    // Get current statistics - using a struct instead of variant map
    struct CurrentStats {
        float mean;
        float median;
        float mode;
        int total_detections;
        int frames_without_person;
        bool person_detected;
    };
    
    CurrentStats get_current_stats() {
        Stats stats = calculate_statistics(detection_ratios);
        
        CurrentStats result;
        result.mean = stats["mean"];
        result.median = stats["median"];
        result.mode = stats["mode"];
        result.total_detections = static_cast<int>(detection_ratios.size());
        result.frames_without_person = frames_without_person;
        result.person_detected = person_detected;
        
        return result;
    }
    
    // Alternative: get stats as map (if you really need string keys)
    map<string, StatValue> get_current_stats_map() {
        Stats float_stats = calculate_statistics(detection_ratios);
        
        map<string, StatValue> result;
        result["mean"] = float_stats["mean"];
        result["median"] = float_stats["median"];
        result["mode"] = float_stats["mode"];
        result["total_detections"] = static_cast<int>(detection_ratios.size());
        result["frames_without_person"] = frames_without_person;
        result["person_detected"] = person_detected;
        
        return result;
    }
    
    // Utility methods
    void clear_ratios() {
        detection_ratios.clear();
    }
    
    size_t get_ratio_count() const {
        return detection_ratios.size();
    }
    
    bool is_person_detected() const {
        return person_detected;
    }
};
int main() {
    // Create configuration
    Config config = {
        {"welcome_threshold", 20.0f},
        {"departure_threshold", 18.0f},
        {"median_ratio_threshold", 30.0f},
        {"mode_ratio_threshold", 20.0f},
        {"arrival_time_seconds", 2.0f},
        {"departure_time_seconds", 3.0f}
    };
    
    // Create tracker
    PersonDetectionTracker tracker(config);
    
    // Example usage
    vector<Box> boxes = {{100.0f, 100.0f, 200.0f, 300.0f}}; // x_min, y_min, x_max, y_max
    int frame_width = 640;
    int frame_height = 480;
    
    auto [detected, action] = tracker.update_detection(boxes, frame_width, frame_height);
    
    cout << "Person detected: " << detected << ", Action: " << action << endl;
    
    return 0;
}