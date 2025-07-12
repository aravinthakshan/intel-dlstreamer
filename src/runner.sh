
Generated bash
#!/bin/bash

# Intel DL Streamer CPU Performance Benchmark Script
# This script benchmarks video inference performance on Intel CPU hardware
# and generates a comprehensive performance report with automatic model/data downloads

# Remove strict error handling for better control
# set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results"
LOG_DIR="${RESULTS_DIR}/logs"
MODELS_DIR="${SCRIPT_DIR}/models"
VIDEOS_DIR="${SCRIPT_DIR}/videos"
REPORTS_DIR="${RESULTS_DIR}/reports"

# Docker configuration
DOCKER_IMAGE="intel/dlstreamer:latest"
CONTAINER_NAME="dlstreamer_benchmark"

# Test configurations
INFERENCE_INTERVALS=(1)
PARALLEL_STREAMS=(1 2 4 8 16)
TEST_DURATION=60  # seconds per test

# Models to test
declare -A MODELS=(
    ["person-vehicle-bike-detection-2004"]="intel/person-vehicle-bike-detection-2004/FP16/person-vehicle-bike-detection-2004.xml"
    ["face-detection-adas-0001"]="intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
    ["vehicle-detection-adas-0002"]="intel/vehicle-detection-adas-0002/FP16/vehicle-detection-adas-0002.xml"
    ["person-detection-retail-0013"]="intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml"
)

# Classification models
declare -A CLASSIFICATION_MODELS=(
    ["vehicle-attributes-recognition-barrier-0039"]="intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml"
    ["person-attributes-recognition-crossroad-0230"]="intel/person-attributes-recognition-crossroad-0230/FP16/person-attributes-recognition-crossroad-0230.xml"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local message="$1"
    local timestamp="[$(date '+%Y-%m-%d %H:%M:%S')]"
    echo -e "${GREEN}${timestamp}${NC} $message"
    
    if [[ -d "$LOG_DIR" ]]; then
        echo -e "${timestamp} $message" >> "${LOG_DIR}/benchmark.log"
    fi
}

error() {
    local message="$1"
    local timestamp="[ERROR]"
    echo -e "${RED}${timestamp}${NC} $message"
    
    if [[ -d "$LOG_DIR" ]]; then
        echo -e "${timestamp} $message" >> "${LOG_DIR}/benchmark.log"
    fi
}

warn() {
    local message="$1"
    local timestamp="[WARNING]"
    echo -e "${YELLOW}${timestamp}${NC} $message"
    
    if [[ -d "$LOG_DIR" ]]; then
        echo -e "${timestamp} $message" >> "${LOG_DIR}/benchmark.log"
    fi
}

info() {
    local message="$1"
    local timestamp="[INFO]"
    echo -e "${BLUE}${timestamp}${NC} $message"
    
    if [[ -d "$LOG_DIR" ]]; then
        echo -e "${timestamp} $message" >> "${LOG_DIR}/benchmark.log"
    fi
}

# Safe directory creation function
create_directory() {
    local dir_path="$1"
    
    if [[ ! -d "$dir_path" ]]; then
        # MODIFICATION: Removed error hiding so you can see permission issues.
        if mkdir -p "$dir_path"; then
            info "Created directory: $dir_path"
        else
            warn "Failed to create directory: $dir_path. Trying with sudo..."
            if sudo mkdir -p "$dir_path"; then
                sudo chown -R "$USER:$USER" "$dir_path"
                info "Created directory with sudo: $dir_path"
            else
                error "Failed to create directory even with sudo: $dir_path"
                return 1
            fi
        fi
    else
        info "Directory already exists: $dir_path"
    fi
    return 0
}

# Create directory structure
setup_directories() {
    log "Setting up directory structure..."
    create_directory "$RESULTS_DIR" || return 1
    create_directory "$LOG_DIR" || return 1
    create_directory "$MODELS_DIR" || return 1
    create_directory "$VIDEOS_DIR" || return 1
    create_directory "$REPORTS_DIR" || return 1
    
    for model in "${!MODELS[@]}"; do
        create_directory "${MODELS_DIR}/intel/${model}/FP16" || warn "Failed to create model directory for $model"
    done
    
    for model in "${!CLASSIFICATION_MODELS[@]}"; do
        create_directory "${MODELS_DIR}/intel/${model}/FP16" || warn "Failed to create classification model directory for $model"
    done
    
    log "Directory structure setup completed"
}

# Safe package installation
install_packages() {
    log "Installing required packages..."
    # MODIFICATION: Removed error hiding for better debugging.
    if ! sudo apt-get update; then
        warn "Initial apt update failed, trying to fix..."
        sudo apt-get update --fix-missing || warn "Could not fix apt sources"
    fi
    
    local packages=("wget" "gnupg" "curl" "jq" "bc")
    for package in "${packages[@]}"; do
        if ! command -v "$package" &> /dev/null; then
            info "Installing $package..."
            if sudo apt-get install -y "$package"; then
                info "Successfully installed $package"
            else
                warn "Failed to install $package, continuing..."
            fi
        else
            info "$package is already installed"
        fi
    done
}

# Download OpenVINO models
download_models() {
    log "Checking and downloading OpenVINO models..."
    
    if ! command -v omz_downloader &> /dev/null; then
        warn "omz_downloader not found. Attempting to install OpenVINO toolkit..."
        install_openvino
    fi
    
    for model in "${!MODELS[@]}"; do
        model_path="${MODELS_DIR}/${MODELS[$model]}"
        if [[ ! -f "$model_path" ]]; then
            log "Downloading model: $model"
            # MODIFICATION: Removed error hiding. Now you will see download errors.
            if command -v omz_downloader &> /dev/null; then
                omz_downloader --name "$model" --output_dir "${MODELS_DIR}" --precision FP16 || {
                    warn "Failed to download $model with omz_downloader, trying alternative..."
                    download_model_alternative "$model"
                }
            else
                download_model_alternative "$model"
            fi
        else
            info "Model $model already exists"
        fi
    done
    
    for model in "${!CLASSIFICATION_MODELS[@]}"; do
        model_path="${MODELS_DIR}/${CLASSIFICATION_MODELS[$model]}"
        if [[ ! -f "$model_path" ]]; then
            log "Downloading classification model: $model"
            if command -v omz_downloader &> /dev/null; then
                omz_downloader --name "$model" --output_dir "${MODELS_DIR}" --precision FP16 || {
                    warn "Failed to download $model with omz_downloader, trying alternative..."
                    download_model_alternative "$model"
                }
            else
                download_model_alternative "$model"
            fi
        else
            info "Classification model $model already exists"
        fi
    done
}

# Install OpenVINO toolkit
install_openvino() {
    log "Installing OpenVINO toolkit..."
    # MODIFICATION: Removed error hiding from key commands.
    wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O /tmp/intel-key.pub
    sudo apt-key add /tmp/intel-key.pub
    rm /tmp/intel-key.pub
    
    local ubuntu_version=$(lsb_release -rs | cut -d. -f1)
    local repo_version="ubuntu20"
    if [[ "$ubuntu_version" == "22" ]]; then repo_version="ubuntu22"; fi
    if [[ "$ubuntu_version" == "24" ]]; then repo_version="ubuntu24"; fi
    
    echo "deb https://apt.repos.intel.com/openvino/2025 $repo_version main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2025.list
    
    sudo apt-get update
    if sudo apt-get install -y openvino-2025.0.0; then
        if [[ -f /opt/intel/openvino_2025/setupvars.sh ]]; then
            source /opt/intel/openvino_2025/setupvars.sh
            info "OpenVINO installed successfully"
            return 0
        fi
    fi
    
    warn "OpenVINO installation failed"
    return 1
}

# Alternative model download method
download_model_alternative() {
    local model_name=$1
    log "Attempting alternative download for $model_name..."
    local base_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"
    local model_dir="${MODELS_DIR}/intel/${model_name}/FP16"
    create_directory "$model_dir"
    
    # MODIFICATION: Removed error hiding from wget.
    if wget -O "${model_dir}/${model_name}.xml" "${base_url}/${model_name}/FP16/${model_name}.xml"; then
        info "Downloaded XML for $model_name"
    else
        warn "Failed to download XML for $model_name"
    fi
    
    if wget -O "${model_dir}/${model_name}.bin" "${base_url}/${model_name}/FP16/${model_name}.bin"; then
        info "Downloaded BIN for $model_name"
    else
        warn "Failed to download BIN for $model_name"
    fi
}

# MODIFICATION: This function now just checks for the required video file.
check_for_video() {
    log "Checking for sample video..."
    local video_file="${VIDEOS_DIR}/person-bicycle-car-detection.mp4"

    if [[ ! -f "$video_file" ]]; then
        error "Video file not found: $video_file"
        error "Please place your 'person-bicycle-car-detection.mp4' inside the 'videos' directory and run again."
        exit 1
    else
        info "Video file found: $video_file"
    fi
}

# Setup Docker environment
setup_docker() {
    log "Setting up Docker environment..."
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker service."
        return 1
    fi
    
    log "Pulling Docker image: $DOCKER_IMAGE"
    if docker pull "$DOCKER_IMAGE"; then
        info "Successfully pulled Docker image"
    else
        error "Failed to pull Docker image: $DOCKER_IMAGE"
        return 1
    fi
    
    export MODELS_PATH="$MODELS_DIR"
    export VIDEOS_PATH="$VIDEOS_DIR"
    return 0
}

run_benchmark() {
    local model_name=$1
    local classification_model=$2
    local inference_interval=$3
    local num_streams=$4
    local device="CPU"
    
    log "Running benchmark: Model=$model_name, Classification=$classification_model, Interval=$inference_interval, Streams=$num_streams, Device=$device"
    
    # --- Define paths and files ---
    local model_path="/home/dlstreamer/models/${MODELS[$model_name]}"
    local video_path="/home/dlstreamer/videos/person-bicycle-car-detection.mp4"
    local result_file="${RESULTS_DIR}/benchmark_${model_name}_${classification_model}_${inference_interval}_${num_streams}_${device}.json"
    local log_file="${LOG_DIR}/benchmark_${model_name}_${classification_model}_${inference_interval}_${num_streams}_${device}.log"

    # --- Safely build the GStreamer pipeline string piece-by-piece ---
    # This prevents syntax errors from empty variables.

    # 1. Define the core pipeline for a single stream
    local core_pipeline="filesrc location=$video_path ! decodebin ! videoconvert ! gvadetect model=$model_path device=$device inference-interval=$inference_interval"

    # 2. Conditionally add the classification element IF it exists
    if [[ -n "$classification_model" ]]; then
        local classification_path="/home/dlstreamer/models/${CLASSIFICATION_MODELS[$classification_model]}"
        # IMPORTANT: Append the classifier to the core pipeline string
        core_pipeline+=" ! gvaclassify model=$classification_path device=$device object-class=vehicle reclassify-interval=$inference_interval"
    fi

    # 3. Define the final elements for measurement
    local sink_pipeline="gvafpscounter ! gvametaconvert format=json ! gvametapublish method=file file-path=/tmp/results.json ! fakesink sync=false"

    # 4. Construct the final command based on the number of streams
    local final_pipeline_cmd=""
    if [[ $num_streams -eq 1 ]]; then
        # For a single stream, the command is simple
        final_pipeline_cmd="gst-launch-1.0 $core_pipeline ! $sink_pipeline"
    else
        # For multiple streams, launch N-1 in the background without measurement
        for ((i=1; i<num_streams; i++)); do
            final_pipeline_cmd+="gst-launch-1.0 $core_pipeline ! fakesink sync=false & "
        done
        # Launch the final stream in the foreground with measurement
        final_pipeline_cmd+="gst-launch-1.0 $core_pipeline ! $sink_pipeline"
    fi

    # --- Execute the benchmark ---
    local start_time=$(date +%s)
    
    # Execute the command inside the Docker container
    timeout ${TEST_DURATION}s docker run --rm \
        -v "${MODELS_DIR}:/home/dlstreamer/models" \
        -v "${VIDEOS_DIR}:/home/dlstreamer/videos" \
        -v "${RESULTS_DIR}:/tmp/results" \
        "$DOCKER_IMAGE" \
        bash -c "$final_pipeline_cmd" > "$log_file" 2>&1
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # --- Extract and save results ---
    local fps=$(grep -oP 'current: \K[0-9.]+' "$log_file" | tail -1 || echo "0")
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    
    cat > "$result_file" << EOF
{
    "model": "$model_name",
    "classification_model": "$classification_model",
    "inference_interval": $inference_interval,
    "num_streams": $num_streams,
    "device": "$device",
    "duration": $duration,
    "fps": $fps,
    "cpu_usage": "$cpu_usage",
    "memory_usage": "$memory_usage",
    "timestamp": "$(date -Iseconds)"
}
EOF
    
    log "Benchmark completed: FPS=$fps, CPU=$cpu_usage%, Memory=$memory_usage%"
}


# Run all benchmarks
run_all_benchmarks() {
    log "Starting comprehensive benchmark suite..."
    # The rest of the script from here down does not need modification.
    local total_tests=0
    local completed_tests=0
    
    for model in "${!MODELS[@]}"; do
        for classification_model in "" "${!CLASSIFICATION_MODELS[@]}"; do
            for interval in "${INFERENCE_INTERVALS[@]}"; do
                for streams in "${PARALLEL_STREAMS[@]}"; do
                    ((total_tests++))
                done
            done
        done
    done
    
    log "Total tests to run: $total_tests"
    
    for model in "${!MODELS[@]}"; do
        for classification_model in "" "${!CLASSIFICATION_MODELS[@]}"; do
            for interval in "${INFERENCE_INTERVALS[@]}"; do
                for streams in "${PARALLEL_STREAMS[@]}"; do
                    ((completed_tests++))
                    info "Progress: $completed_tests/$total_tests"
                    run_benchmark "$model" "$classification_model" "$interval" "$streams"
                    sleep 2
                done
            done
        done
    done
    
    log "All benchmarks completed!"
}

# Analyze results
analyze_results() {
    log "Analyzing benchmark results..."
    local analysis_file="${REPORTS_DIR}/performance_analysis.json"
    local best_fps=0
    local best_config=""
    local max_streams=0
    
    for result_file in "${RESULTS_DIR}"/benchmark_*.json; do
        if [[ -f "$result_file" ]]; then
            local fps=$(jq -r '.fps' "$result_file" 2>/dev/null || echo "0")
            local streams=$(jq -r '.num_streams' "$result_file" 2>/dev/null || echo "0")
            
            if (( $(echo "$fps > $best_fps" | bc -l) )); then
                best_fps=$fps
                best_config=$(basename "$result_file" .json)
            fi
            
            if (( streams > max_streams )); then
                max_streams=$streams
            fi
        fi
    done
    
    cat > "$analysis_file" << EOF
{
    "analysis_timestamp": "$(date -Iseconds)",
    "best_performance": {
        "fps": $best_fps,
        "configuration": "$best_config"
    },
    "maximum_streams": $max_streams,
    "total_tests": $(ls -1 "${RESULTS_DIR}"/benchmark_*.json 2>/dev/null | wc -l),
    "bottleneck_analysis": "$(determine_bottleneck)"
}
EOF
    
    log "Analysis completed. Best FPS: $best_fps, Max streams: $max_streams"
}

# Determine system bottleneck
determine_bottleneck() {
    local cpu_avg=0
    local mem_avg=0
    local count=0
    for result_file in "${RESULTS_DIR}"/benchmark_*.json; do
        if [[ -f "$result_file" ]]; then
            cpu_avg=$(echo "$cpu_avg + $(jq -r '.cpu_usage' "$result_file")" | bc)
            mem_avg=$(echo "$mem_avg + $(jq -r '.memory_usage' "$result_file")" | bc)
            ((count++))
        fi
    done
    if [[ $count -gt 0 ]]; then
        cpu_avg=$(echo "scale=2; $cpu_avg / $count" | bc)
        if (( $(echo "$cpu_avg > 80" | bc -l) )); then
            echo "CPU"
        else
            echo "IO/Other"
        fi
    else
        echo "N/A"
    fi
}

# Generate comprehensive report
generate_report() {
    log "Generating comprehensive performance report..."
    local report_file="${REPORTS_DIR}/Intel_DL_Streamer_Performance_Report.md"
    
    # ... (The report generation part is long and doesn't need changes)
    cat > "$report_file" << EOF
# Intel DL Streamer Performance Analysis Report
Generated on $(date)

## Executive Summary
This report details the performance of Intel DL Streamer on this system's CPU using various models and stream counts against the \`person-bicycle-car-detection.mp4\` video file.

## Test Configuration
- **Hardware**: CPU
- **Video Source**: \`person-bicycle-car-detection.mp4\`
- **Inference Intervals**: ${INFERENCE_INTERVALS[*]}
- **Parallel Streams**: ${PARALLEL_STREAMS[*]}

## Performance Summary
| Metric | Value |
|--------|-------|
EOF
    
    local analysis_data=$(cat "${REPORTS_DIR}/performance_analysis.json")
    local best_fps=$(echo "$analysis_data" | jq -r '.best_performance.fps')
    local best_config=$(echo "$analysis_data" | jq -r '.best_performance.configuration')
    local max_streams=$(echo "$analysis_data" | jq -r '.maximum_streams')
    local bottleneck=$(echo "$analysis_data" | jq -r '.bottleneck_analysis')
    
    echo "| **Peak FPS** | \`$best_fps\` |" >> "$report_file"
    echo "| **Best Configuration** | \`$best_config\` |" >> "$report_file"
    echo "| **Max Streams Tested** | \`$max_streams\` |" >> "$report_file"
    echo "| **Primary Bottleneck** | \`$bottleneck\` |" >> "$report_file"
    
    echo -e "\n## Detailed Results\n" >> "$report_file"
    echo "| Model | Classification | Interval | Streams | FPS | CPU % | Memory % |" >> "$report_file"
    echo "|-------|----------------|----------|---------|-----|-------|----------|" >> "$report_file"
    
    for result_file in "${RESULTS_DIR}"/benchmark_*.json; do
        if [[ -f "$result_file" ]]; then
            local model=$(jq -r '.model' "$result_file")
            local class=$(jq -r '.classification_model' "$result_file" | sed 's/""/N\/A/')
            local interval=$(jq -r '.inference_interval' "$result_file")
            local streams=$(jq -r '.num_streams' "$result_file")
            local fps=$(jq -r '.fps' "$result_file")
            local cpu=$(jq -r '.cpu_usage' "$result_file")
            local mem=$(jq -r '.memory_usage' "$result_file")
            printf "| %-38s | %-28s | %-8s | %-7s | %-5s | %-5s | %-8s |\n" "$model" "$class" "$interval" "$streams" "$fps" "$cpu" "$mem" >> "$report_file"
        fi
    done
    
    log "Comprehensive report generated: ${report_file}"
}

# ---
# Utility: Download and Convert YOLO Models for DL Streamer
# Usage: ./src/runner.sh download_yolo_models
# ---
download_yolo_models() {
    log "Downloading YOLO models using OpenVINO OMZ..."
    local yolo_models=(
        "yolo-v3-tf"
        "yolo-v4-tf"
    )
    for model in "${yolo_models[@]}"; do
        omz_downloader --name "$model" --output_dir "${MODELS_DIR}" --precision FP16 || {
            warn "Failed to download $model"; continue;
        }
        omz_converter --name "$model" --download_dir "${MODELS_DIR}" --output_dir "${MODELS_DIR}" --precision FP16 || {
            warn "Failed to convert $model"; continue;
        }
        info "Downloaded and converted $model to IR format."
    done
    log "YOLO models are ready in ${MODELS_DIR} for DL Streamer."
}

# Add command-line utility support
if [[ "$1" == "download_yolo_models" ]]; then
    setup_directories
    install_packages
    download_yolo_models
    exit 0
fi

# Main execution function
main() {
    log "Starting Intel DL Streamer Benchmark Script"
    
    setup_directories
    install_packages
    # MODIFICATION: Replaced video download with a check for the local file.
    check_for_video
    download_models
    setup_docker
    
    run_all_benchmarks
    
    analyze_results
    generate_report
    
    log "Script finished successfully!"
    info "Find the detailed report here: ${REPORTS_DIR}/Intel_DL_Streamer_Performance_Report.md"
    info "Find all result data and logs here: ${RESULTS_DIR}"
}

# Run the main function
main
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
How to Run