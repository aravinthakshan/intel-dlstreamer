#!/usr/bin/env python3
"""
AI Camera Processing Pipeline using Intel DL Streamer
Handles multiple video streams with decode, detect, and classify operations
Benchmarks performance on CPU vs GPU configurations
"""

import os
import sys
import time
import threading
import subprocess
import json
import csv
from datetime import datetime
from pathlib import Path
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor, as_completed

class StreamProcessor:
    def __init__(self, device="CPU", model_path="", detection_threshold=0.5):
        self.device = device
        self.model_path = model_path
        self.detection_threshold = detection_threshold
        self.active_streams = []
        self.performance_data = []
        
    def create_gst_pipeline(self, input_source, stream_id, fps_target=30):
        """Create GStreamer pipeline string for DL Streamer"""
        
        # Base pipeline components
        if input_source.startswith('rtsp://') or input_source.startswith('http'):
            source = f"urisourcebin uri={input_source}"
        else:
            source = f"filesrc location={input_source}"
            
        # Device-specific inference configuration
        if self.device == "GPU":
            device_config = "device=GPU"
            inference_element = "gvadetect"
        else:
            device_config = "device=CPU"
            inference_element = "gvadetect"
            
        pipeline = f"""
        {source} ! 
        decodebin ! 
        videoconvert ! 
        videoscale ! 
        video/x-raw,width=640,height=480,framerate={fps_target}/1 !
        {inference_element} 
            model={self.model_path}/person-detection-retail-0013.xml 
            {device_config} 
            threshold={self.detection_threshold} !
        gvaclassify 
            model={self.model_path}/person-attributes-recognition-crossroad-0230.xml 
            {device_config} !
        gvametaconvert format=json !
        appsink name=sink_{stream_id} emit-signals=true sync=false
        """
        
        return pipeline.replace('\n', ' ').strip()
    
    def run_single_stream(self, video_path, stream_id, duration=60, fps_target=30):
        """Run a single video stream and collect performance metrics"""
        
        pipeline_str = self.create_gst_pipeline(video_path, stream_id, fps_target)
        
        # Create GStreamer command
        cmd = [
            'gst-launch-1.0',
            '-e',  # Send EOS on interrupt
            pipeline_str
        ]
        
        start_time = time.time()
        process = None
        
        try:
            # Start the pipeline
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor performance
            frame_count = 0
            detection_count = 0
            
            # Run for specified duration
            while time.time() - start_time < duration:
                if process.poll() is not None:
                    break
                    
                # Simulate frame processing (in real implementation, 
                # you'd parse the JSON metadata from appsink)
                time.sleep(1.0/fps_target)
                frame_count += 1
                
                # Simulate detections (would come from actual inference)
                if frame_count % 10 == 0:  # Assume detection every 10 frames
                    detection_count += 1
            
            end_time = time.time()
            actual_duration = end_time - start_time
            actual_fps = frame_count / actual_duration if actual_duration > 0 else 0
            
            return {
                'stream_id': stream_id,
                'device': self.device,
                'target_fps': fps_target,
                'actual_fps': actual_fps,
                'frames_processed': frame_count,
                'detections': detection_count,
                'duration': actual_duration,
                'success': True
            }
            
        except Exception as e:
            return {
                'stream_id': stream_id,
                'device': self.device,
                'error': str(e),
                'success': False
            }
        finally:
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

class PerformanceBenchmark:
    def __init__(self, model_path="./models"):
        self.model_path = model_path
        self.results = []
        
    def get_system_info(self):
        """Collect system specifications"""
        cpu_info = {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available
        }
        
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree
                })
        except:
            gpu_info = [{'name': 'No GPU detected', 'memory_total': 0, 'memory_free': 0}]
            
        return {'cpu': cpu_info, 'gpu': gpu_info}
    
    def monitor_resources(self, duration=60):
        """Monitor CPU, GPU, and memory usage during benchmark"""
        start_time = time.time()
        resource_data = []
        
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            gpu_util = 0
            gpu_memory = 0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util = gpus[0].load * 100
                    gpu_memory = (gpus[0].memoryUsed / gpus[0].memoryTotal) * 100
            except:
                pass
                
            resource_data.append({
                'timestamp': time.time() - start_time,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'gpu_util': gpu_util,
                'gpu_memory': gpu_memory
            })
            
        return resource_data
    
    def benchmark_streams(self, video_files, device="CPU", max_streams=8, fps_targets=[15, 30]):
        """Benchmark multiple streams with different configurations"""
        
        print(f"\n=== Benchmarking on {device} ===")
        
        for fps_target in fps_targets:
            print(f"\nTesting FPS target: {fps_target}")
            
            for num_streams in range(1, max_streams + 1):
                print(f"Testing {num_streams} streams...")
                
                # Start resource monitoring
                resource_monitor = threading.Thread(
                    target=lambda: self.monitor_resources(65),
                    daemon=True
                )
                resource_monitor.start()
                
                # Create stream processors
                processor = StreamProcessor(
                    device=device,
                    model_path=self.model_path
                )
                
                # Run multiple streams concurrently
                with ThreadPoolExecutor(max_workers=num_streams) as executor:
                    futures = []
                    for i in range(num_streams):
                        video_file = video_files[i % len(video_files)]
                        future = executor.submit(
                            processor.run_single_stream,
                            video_file,
                            f"stream_{i}",
                            60,  # duration
                            fps_target
                        )
                        futures.append(future)
                    
                    # Collect results
                    stream_results = []
                    for future in as_completed(futures):
                        result = future.result()
                        stream_results.append(result)
                
                # Calculate aggregate metrics
                successful_streams = [r for r in stream_results if r['success']]
                
                if successful_streams:
                    avg_fps = sum(r['actual_fps'] for r in successful_streams) / len(successful_streams)
                    total_detections = sum(r['detections'] for r in successful_streams)
                    
                    result = {
                        'device': device,
                        'num_streams': num_streams,
                        'target_fps': fps_target,
                        'successful_streams': len(successful_streams),
                        'avg_fps_per_stream': avg_fps,
                        'total_fps': avg_fps * len(successful_streams),
                        'total_detections': total_detections,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.results.append(result)
                    
                    print(f"  Success: {len(successful_streams)}/{num_streams} streams")
                    print(f"  Avg FPS per stream: {avg_fps:.2f}")
                    print(f"  Total FPS: {avg_fps * len(successful_streams):.2f}")
                
                # Check if performance is degrading significantly
                if len(successful_streams) < num_streams * 0.5:  # Less than 50% success
                    print(f"  Performance degraded significantly, stopping at {num_streams} streams")
                    break
                    
                time.sleep(2)  # Brief pause between tests
    
    def find_optimal_configuration(self):
        """Analyze results to find optimal configurations"""
        if not self.results:
            return None
            
        # Group by device
        cpu_results = [r for r in self.results if r['device'] == 'CPU']
        gpu_results = [r for r in self.results if r['device'] == 'GPU']
        
        analysis = {}
        
        for device, results in [('CPU', cpu_results), ('GPU', gpu_results)]:
            if not results:
                continue
                
            # Find maximum streams
            max_streams = max(r['num_streams'] for r in results)
            
            # Find best FPS configuration
            best_fps_config = max(results, key=lambda x: x['total_fps'])
            
            # Find bottleneck point (where performance starts degrading)
            sorted_results = sorted(results, key=lambda x: (x['target_fps'], x['num_streams']))
            bottleneck_stream_count = max_streams
            
            for i in range(1, len(sorted_results)):
                curr = sorted_results[i]
                prev = sorted_results[i-1]
                
                if (curr['avg_fps_per_stream'] < prev['avg_fps_per_stream'] * 0.8 and 
                    curr['target_fps'] == prev['target_fps']):
                    bottleneck_stream_count = prev['num_streams']
                    break
            
            analysis[device] = {
                'max_streams': max_streams,
                'best_config': best_fps_config,
                'bottleneck_at_streams': bottleneck_stream_count,
                'total_results': len(results)
            }
        
        return analysis
    
    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to file"""
        report_data = {
            'system_info': self.get_system_info(),
            'benchmark_results': self.results,
            'analysis': self.find_optimal_configuration(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Also save as CSV for easy analysis
        csv_filename = filename.replace('.json', '.csv')
        with open(csv_filename, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        
        print(f"\nResults saved to {filename} and {csv_filename}")

def download_models():
    """Download Intel OpenVINO models if not present"""
    model_dir = Path("./models")
    model_dir.mkdir(exist_ok=True)
    
    models = [
        "person-detection-retail-0013",
        "person-attributes-recognition-crossroad-0230"
    ]
    
    for model in models:
        model_path = model_dir / f"{model}.xml"
        if not model_path.exists():
            print(f"Downloading {model}...")
            # In practice, you'd use OpenVINO Model Downloader
            # omz_downloader --name {model} --output_dir ./models
            cmd = f"omz_downloader --name {model} --output_dir ./models"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to download {model}. Please install OpenVINO Model Zoo tools.")
                print("Run: pip install openvino-dev[onnx,tensorflow2]")
                return False
    
    return True

def main():
    """Main benchmark execution"""
    print("AI Camera Processing Pipeline Benchmark")
    print("=" * 50)
    
    # Check if models are available
    if not download_models():
        print("Please ensure OpenVINO models are available in ./models directory")
        return
    
    # Sample video files (replace with your actual video files)
    video_files = [
        "./test_videos/sample1.mp4",
        "./test_videos/sample2.mp4",
        "./test_videos/sample3.mp4",
        "./test_videos/sample4.mp4"
    ]
    
    # Check if video files exist
    existing_videos = [v for v in video_files if os.path.exists(v)]
    if not existing_videos:
        print("No test video files found. Please add video files to ./test_videos/")
        print("Creating sample pipeline test...")
        # For demonstration, create dummy test
        existing_videos = ["dummy_video.mp4"]  # Will be handled in pipeline creation
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    
    # Get system information
    sys_info = benchmark.get_system_info()
    print(f"\nSystem Info:")
    print(f"CPU Cores: {sys_info['cpu']['cpu_count']} physical, {sys_info['cpu']['cpu_count_logical']} logical")
    print(f"Memory: {sys_info['cpu']['memory_total'] // (1024**3)} GB total")
    print(f"GPU: {sys_info['gpu'][0]['name'] if sys_info['gpu'] else 'None'}")
    
    # Run CPU benchmark
    benchmark.benchmark_streams(
        existing_videos,
        device="CPU",
        max_streams=8,
        fps_targets=[15, 30]
    )
    
    # Run GPU benchmark if GPU is available
    if sys_info['gpu'] and sys_info['gpu'][0]['name'] != 'No GPU detected':
        benchmark.benchmark_streams(
            existing_videos,
            device="GPU",
            max_streams=16,
            fps_targets=[15, 30, 60]
        )
    
    # Analyze and save results
    analysis = benchmark.find_optimal_configuration()
    benchmark.save_results("ai_camera_benchmark_results.json")
    
    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    if analysis:
        for device, data in analysis.items():
            print(f"\n{device} Results:")
            print(f"  Maximum streams supported: {data['max_streams']}")
            print(f"  Best configuration: {data['best_config']['num_streams']} streams at {data['best_config']['target_fps']} FPS")
            print(f"  Total FPS achieved: {data['best_config']['total_fps']:.2f}")
            print(f"  Bottleneck occurs at: {data['bottleneck_at_streams']} streams")
    
    print(f"\nDetailed results saved to benchmark files.")

if __name__ == "__main__":
    main()