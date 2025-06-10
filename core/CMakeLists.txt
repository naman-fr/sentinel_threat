cmake_minimum_required(VERSION 3.18)
project(sentinel_core LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)

# Set CUDA specific flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add source files
set(SOURCES
    ThreatDetectionEngine.cpp
    CudaStreamManager.cpp
    CircularBuffer.cpp
    ThreadPool.cpp
)

# Add header files
set(HEADERS
    ThreatDetectionEngine.hpp
    CudaStreamManager.hpp
    CircularBuffer.hpp
    ThreadPool.hpp
)

# Create library
add_library(sentinel_core SHARED ${SOURCES} ${HEADERS})

# Set include directories
target_include_directories(sentinel_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(sentinel_core
    PUBLIC
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        spdlog::spdlog
        Threads::Threads
)

# Install rules
install(TARGETS sentinel_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS}
    DESTINATION include/sentinel/core
)

# Enable testing
enable_testing()
add_subdirectory(tests)

# Set compiler warnings
if(MSVC)
    target_compile_options(sentinel_core PRIVATE /W4)
else()
    target_compile_options(sentinel_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Add custom target for formatting
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT}
        -i
        ${SOURCES}
        ${HEADERS}
        COMMENT "Formatting source files"
    )
endif() 