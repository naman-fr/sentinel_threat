#include "ThreatDetectionEngine.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <chrono>
#include <spdlog/spdlog.h>

namespace sentinel {
namespace core {

ThreatDetectionEngine::ThreatDetectionEngine() {
    try {
        initializeCUDA();
        initializeThreadPool();
        is_processing_ = false;
        metrics_ = {0.0, 0, 0};
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize ThreatDetectionEngine: {}", e.what());
        throw;
    }
}

ThreatDetectionEngine::~ThreatDetectionEngine() {
    is_processing_ = false;
    // Cleanup will be handled by smart pointers
}

void ThreatDetectionEngine::initializeCUDA() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        throw std::runtime_error("No CUDA-capable devices found");
    }

    // Initialize CUDA streams for parallel processing
    gpu_streams_ = std::make_unique<CudaStreamManager>();
    
    // Set device properties for optimal performance
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaSetDevice(0);
    
    spdlog::info("Initialized CUDA on device: {}", prop.name);
}

void ThreatDetectionEngine::initializeThreadPool() {
    const size_t num_threads = std::thread::hardware_concurrency();
    processing_pool_ = std::make_unique<ThreadPool>(num_threads);
    spdlog::info("Initialized thread pool with {} threads", num_threads);
}

ThreatLevel ThreatDetectionEngine::processVideoStream(const VideoFrame& frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!is_processing_) {
        throw std::runtime_error("Engine is not in processing state");
    }

    std::lock_guard<std::mutex> lock(processing_mutex_);
    
    try {
        // Process frame on GPU
        processFrameGPU(frame);
        
        // Calculate threat level
        ThreatLevel threat;
        threat.confidence = calculateThreatConfidence(frame);
        threat.timestamp = frame.timestamp;
        
        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        metrics_.average_latency = (metrics_.average_latency * metrics_.processed_frames + latency) 
                                 / (metrics_.processed_frames + 1);
        metrics_.processed_frames++;
        
        if (threat.confidence > 0.8f) {
            metrics_.detected_threats++;
            spdlog::warn("High confidence threat detected: {}%", threat.confidence * 100);
        }
        
        return threat;
    } catch (const std::exception& e) {
        spdlog::error("Error processing video frame: {}", e.what());
        throw;
    }
}

void ThreatDetectionEngine::processAudioSignals(const AudioBatch& batch) {
    if (!is_processing_) {
        throw std::runtime_error("Engine is not in processing state");
    }

    std::lock_guard<std::mutex> lock(processing_mutex_);
    
    try {
        // Process audio on GPU
        processAudioGPU(batch);
        
        // Additional audio analysis
        analyzeAudioPatterns(batch);
        
    } catch (const std::exception& e) {
        spdlog::error("Error processing audio signals: {}", e.what());
        throw;
    }
}

FusedThreatData ThreatDetectionEngine::fuseSensorData(const MultiSensorInput& input) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!is_processing_) {
        throw std::runtime_error("Engine is not in processing state");
    }

    std::lock_guard<std::mutex> lock(processing_mutex_);
    
    try {
        // Process video and audio in parallel
        auto video_future = processing_pool_->enqueue([this, &input]() {
            return processVideoStream(input.video);
        });
        
        auto audio_future = processing_pool_->enqueue([this, &input]() {
            processAudioSignals(input.audio);
        });
        
        // Wait for both processing tasks to complete
        ThreatLevel video_threat = video_future.get();
        audio_future.wait();
        
        // Fuse the results
        FusedThreatData fused_data;
        fused_data.threat_level = video_threat;
        fused_data.timestamp = input.video.timestamp;
        
        // Calculate sensor confidence
        fused_data.sensor_confidence = calculateSensorConfidence(input);
        
        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        spdlog::debug("Sensor fusion completed in {}ms", latency);
        
        return fused_data;
        
    } catch (const std::exception& e) {
        spdlog::error("Error fusing sensor data: {}", e.what());
        throw;
    }
}

void ThreatDetectionEngine::setProcessingThreads(size_t num_threads) {
    processing_pool_ = std::make_unique<ThreadPool>(num_threads);
    spdlog::info("Updated thread pool to {} threads", num_threads);
}

void ThreatDetectionEngine::setBufferSize(size_t size) {
    data_buffer_ = std::make_unique<CircularBuffer<SensorData>>(size);
    spdlog::info("Updated buffer size to {}", size);
}

void ThreatDetectionEngine::enableGPUAcceleration(bool enable) {
    if (enable) {
        initializeCUDA();
    } else {
        gpu_streams_.reset();
    }
    spdlog::info("GPU acceleration {}", enable ? "enabled" : "disabled");
}

} // namespace core
} // namespace sentinel 