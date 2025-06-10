#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>

namespace sentinel {
namespace core {

// Forward declarations
class CudaStreamManager;
class CircularBuffer;
class ThreadPool;

struct VideoFrame {
    uint8_t* data;
    size_t width;
    size_t height;
    size_t channels;
    int64_t timestamp;
};

struct AudioBatch {
    float* samples;
    size_t num_samples;
    size_t num_channels;
    int64_t timestamp;
};

struct MultiSensorInput {
    VideoFrame video;
    AudioBatch audio;
    // Additional sensor data structures
};

struct ThreatLevel {
    float confidence;
    std::string threat_type;
    std::vector<float> bounding_box;
    int64_t timestamp;
};

struct FusedThreatData {
    ThreatLevel threat_level;
    std::vector<float> sensor_confidence;
    std::string fusion_method;
    int64_t timestamp;
};

class ThreatDetectionEngine {
public:
    ThreatDetectionEngine();
    ~ThreatDetectionEngine();

    // Core processing methods
    ThreatLevel processVideoStream(const VideoFrame& frame);
    void processAudioSignals(const AudioBatch& batch);
    FusedThreatData fuseSensorData(const MultiSensorInput& input);

    // Configuration methods
    void setProcessingThreads(size_t num_threads);
    void setBufferSize(size_t size);
    void enableGPUAcceleration(bool enable);

private:
    // CUDA and processing management
    std::unique_ptr<CudaStreamManager> gpu_streams_;
    std::unique_ptr<CircularBuffer<SensorData>> data_buffer_;
    std::unique_ptr<ThreadPool> processing_pool_;

    // Internal processing methods
    void initializeCUDA();
    void initializeThreadPool();
    void processFrameGPU(const VideoFrame& frame);
    void processAudioGPU(const AudioBatch& batch);
    
    // Thread safety
    std::mutex processing_mutex_;
    std::atomic<bool> is_processing_;
    
    // Performance monitoring
    struct PerformanceMetrics {
        double average_latency;
        size_t processed_frames;
        size_t detected_threats;
    } metrics_;
};

} // namespace core
} // namespace sentinel 