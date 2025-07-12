#!/bin/bash
# Script to download Ultralytics YOLOv8n model and convert to OpenVINO IR format

set -e

# Install Ultralytics if not already installed
python3 -m pip install --upgrade ultralytics

# Create model directory
mkdir -p models/yolo
cd models/yolo

# Download YOLOv8n model
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Convert to ONNX
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"

# Source OpenVINO environment (adjust path if needed)
if [ -f /opt/intel/openvino/bin/setupvars.sh ]; then
    source /opt/intel/openvino/bin/setupvars.sh
fi

# Convert ONNX to OpenVINO IR
mo --input_model yolov8n.onnx --output_dir ./openvino_ir

echo "Conversion complete. IR files are in models/yolo/openvino_ir/" 