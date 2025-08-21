#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Tính tỉ lệ diện tích của bounding box so với frame (theo %)
double calculateAreaRatio(const vector<float>& box, int frameWidth, int frameHeight) {
    double boxWidth = box[2] - box[0];
    double boxHeight = box[3] - box[1];
    double boxArea = boxWidth * boxHeight;
    double frameArea = frameWidth * frameHeight;
    return (boxArea / frameArea) * 100.0;
}

// Tính trung bình (mean)
double calculateMean(const vector<double>& ratios) {
    if (ratios.empty()) return 0.0;
    double sum = 0.0;
    for (double val : ratios) sum += val;
    return sum / ratios.size();
}

// Tính trung vị (median)
double calculateMedian(vector<double> ratios) {
    if (ratios.empty()) return 0.0;
    sort(ratios.begin(), ratios.end());
    size_t n = ratios.size();
    if (n % 2 == 0)
        return (ratios[n / 2 - 1] + ratios[n / 2]) / 2.0;
    else
        return ratios[n / 2];
}

// Tính mode (phổ biến nhất) với sai số (tolerance)
double calculateMode(const vector<double>& ratios, double tolerance = 1.0) {
    if (ratios.empty()) return 0.0;
    map<int, int> freq;
    for (double val : ratios) {
        int rounded = static_cast<int>(round(val / tolerance));
        freq[rounded]++;
    }

    int maxCount = 0;
    int modeRounded = 0;
    for (const auto& [key, count] : freq) {
        if (count > maxCount) {
            maxCount = count;
            modeRounded = key;
        }
    }

    return modeRounded * tolerance;
}

// Phân tích arrival/departure dựa trên các ngưỡng
string detectArrivalOrDeparture(
    int humanCount,
    int nonHumanCount,
    bool currentHumanDetected,
    double medianRatio,
    double modeRatio,
    double arrivalMedianThreshold,
    double arrivalModeThreshold,
    double departureMedianThreshold,
    double departureModeThreshold,
    int arrivalFrameThreshold,
    int departureFrameThreshold
) {
    if (humanCount >= arrivalFrameThreshold &&
        !currentHumanDetected &&
        medianRatio > arrivalMedianThreshold &&
        modeRatio > arrivalModeThreshold) {
        return "arrival";
    }

    if (nonHumanCount > departureFrameThreshold &&
        currentHumanDetected &&
        medianRatio < departureMedianThreshold &&
        modeRatio < departureModeThreshold) {
        return "departure";
    }

    return "none";
}

// // Test ví dụ
// int main() {
//     vector<vector<float>> boxes = {
//         {100, 100, 300, 300},
//         {150, 120, 320, 310},
//         {130, 110, 310, 305}
//     };

//     int frameWidth = 640;
//     int frameHeight = 480;
//     vector<double> areaRatios;

//     for (const auto& box : boxes) {
//         double ratio = calculateAreaRatio(box, frameWidth, frameHeight);
//         areaRatios.push_back(ratio);
//     }

//     double meanRatio = calculateMean(areaRatios);
//     double medianRatio = calculateMedian(areaRatios);
//     double modeRatio = calculateMode(areaRatios, 1.0);

//     cout << fixed << setprecision(2);
//     cout << "Mean area ratio: " << meanRatio << "%" << endl;
//     cout << "Median area ratio: " << medianRatio << "%" << endl;
//     cout << "Mode area ratio: " << modeRatio << "%" << endl;

//     string result = detectArrivalOrDeparture(
//         25, // human count
//         5,  // non-human count
//         false, // current human detected
//         medianRatio,
//         modeRatio,
//         20.0, // arrival median threshold
//         15.0, // arrival mode threshold
//         10.0, // departure median threshold
//         8.0,  // departure mode threshold
//         20,   // arrival frame threshold
//         18    // departure frame threshold
//     );

//     cout << "Detected event: " << result << endl;

//     return 0;
// }
