# Intel DL Streamer Benchmark Suite

## Overview

This project provides a comprehensive benchmarking pipeline for evaluating Intel® Deep Learning Streamer (DL Streamer) performance on Intel hardware (CPU and GPU) using Docker. The pipeline decodes video, performs detection and classification, and measures the maximum number of supported camera streams, optimum FPS, and model performance on Intel hardware. It also identifies system bottlenecks (CPU, GPU, or IO) and generates a detailed report.

### Project Outcomes
- **Maximum number of streams, FPS, and best model on CPU**
- **Maximum number of streams, FPS, and best model on GPU**
- **Identification of bottlenecks (CPU, GPU, or IO)**
- **Comprehensive report on pipeline, models, and results**

## Directory Structure

- `src/runner.sh` — Main benchmarking script (run this inside Docker)
- `video/person-bicycle-car-detection.mp4` — Required sample video file
- `annotations/`, `sequences/` — Data directories (not required for benchmarking)
- `benchmark_results/` — Results, logs, and reports (auto-created)

## Prerequisites

- **Docker** installed and running on your system
- **NVIDIA/Intel GPU drivers** (if benchmarking on GPU)
- **X11 GUI access** (for video display, optional)
- **Sample video**: `video/person-bicycle-car-detection.mp4` (must be present)

## 1. Prepare the Environment

### 1.1. Clone the Repository
```bash
git clone <this-repo-url>
cd intel-dl
```

### 1.2. Place the Required Video File
- Download or copy your sample video as `person-bicycle-car-detection.mp4` into the `video/` directory:

```bash
mkdir -p video
cp /path/to/your/person-bicycle-car-detection.mp4 video/
```

> **Note:** The benchmarking script will not run unless this file is present.

### 1.3. Make the Runner Script Executable

```bash
chmod +x src/runner.sh
```

## 2. Docker Setup

### 2.1. Pull the Intel DL Streamer Docker Image

The benchmarking script will automatically pull the latest DL Streamer image, but you can do it manually:

```bash
docker pull intel/dlstreamer:latest
```

### 2.2. Mounting Directories

The benchmarking script mounts the following directories into the Docker container:

- **Models**: `src/models` → `/home/dlstreamer/models`
- **Videos**: `video` → `/home/dlstreamer/videos`
- **Results**: `src/benchmark_results` → `/tmp/results`

This is handled automatically by `src/runner.sh`.

### 2.3. Allowing GUI Access to Docker (for Video Display)

If you want to see the video output (e.g., with `autovideosink`), you must allow Docker containers to access your X11 display:

#### On Linux (X11):

1. Allow local connections to the X server:
   ```bash
   xhost +local:docker
   ```
2. When running Docker manually, add these flags:
   ```bash
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   --device /dev/dri \
   --group-add $(stat -c "%g" /dev/dri/render*)
   ```
   (The benchmarking script will handle device mounting for you.)

> **Note:** If you are running headless or do not need video display, you can ignore this step.

### 2.4. Permissions

- Ensure you have permission to run Docker and access `/dev/dri` for GPU benchmarking.
- If you encounter permission errors with directories, the script will attempt to create them with `sudo` and set ownership to your user.

## 3. Running the Benchmark

All benchmarking is managed by the `src/runner.sh` script. This script:
- Sets up directories and downloads required models (using OpenVINO OMZ)
- Checks for the required video file
- Pulls the Docker image and sets up the environment
- Runs a comprehensive suite of benchmarks (various models, stream counts, and intervals)
- Collects FPS, CPU, and memory usage
- Analyzes results and generates a detailed Markdown report

### To Run:

```bash
./src/runner.sh
```

- All logs, results, and reports will be saved in `src/benchmark_results/`.
- The final report will be in `src/benchmark_results/reports/Intel_DL_Streamer_Performance_Report.md`.

## 4. What the Script Does (Reference)

For full details of the benchmarking process, see [`src/runner.sh`](src/runner.sh). In summary, it:
- Downloads and prepares OpenVINO models
- Runs GStreamer pipelines (via DL Streamer) for detection and classification
- Varies the number of parallel streams and inference intervals
- Measures FPS, CPU, and memory usage
- Analyzes bottlenecks (CPU, IO, etc.)
- Produces a comprehensive Markdown report

## 5. Troubleshooting

- **Video file not found:** Ensure `video/person-bicycle-car-detection.mp4` exists.
- **Docker permission errors:** Add your user to the `docker` group or run with `sudo`.
- **X11 errors:** Make sure you have run `xhost +local:docker` and are not running headless.
- **Model download issues:** The script will attempt alternative download methods if OMZ is unavailable.

## 6. References

- [Intel® DL Streamer Documentation](https://www.intel.com/content/www/us/en/developer/tools/dl-streamer/overview.html)
- [Intel® DL Streamer Official Tutorial: Build Object Detection Pipeline](https://dlstreamer.github.io/get_started/tutorial.html#exercise-1-build-object-detection-pipeline)
- See `referece.txt` for a full tutorial and pipeline examples (from Intel documentation)

---

**Project Report:**
The detailed project report is also included in this repository as `Intel Project Report.pdf`.

## 8. Downloading and Converting Ultralytics YOLO Models with OpenVINO

You can use the following script to download Ultralytics YOLO models (e.g., YOLOv5, YOLOv8) and convert them to OpenVINO IR format for use in the benchmarking pipeline.

### Prerequisites
- Python 3.8+
- [Ultralytics](https://github.com/ultralytics/ultralytics) (for YOLOv8) or [YOLOv5](https://github.com/ultralytics/yolov5)
- [OpenVINO Toolkit](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)

### Example Script: `src/download_and_convert_yolo.sh`

```bash
#!/bin/bash
# Download YOLOv8n model and convert to OpenVINO IR

# Download YOLOv8n (or change to yolov5s.pt for YOLOv5)
python3 -m pip install ultralytics
mkdir -p models/yolo
cd models/yolo

# Download YOLOv8n
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Convert to ONNX
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"

# Convert ONNX to OpenVINO IR
source /opt/intel/openvino/bin/setupvars.sh
mo --input_model yolov8n.onnx --output_dir ./openvino_ir
```

- For YOLOv5, use the [YOLOv5 export instructions](https://github.com/ultralytics/yolov5/wiki/Export-OpenVINO).
- Adjust model names as needed (e.g., `yolov8n.pt`, `yolov8s.pt`, `yolov5s.pt`).

> **Note:** Make sure OpenVINO's Model Optimizer (`mo`) is in your PATH. You may need to adjust the script for your environment.
