🚦 Smart AI Traffic Signal Management System

A perception-based intelligent traffic management system that dynamically adapts signal timings using real-time computer vision and deep learning.

📌 Overview

Traditional traffic signals operate on fixed timers, leading to unnecessary waiting times and poor traffic flow at high-density intersections. This project solves that by building an AI-powered system that detects vehicles in real time and dynamically adjusts green-light durations based on live traffic density — resulting in a 35%+ improvement in estimated intersection throughput.


🎯 Key Features

🔍 Real-time Vehicle Detection — Detects and classifies vehicles across 4 traffic zones using YOLOv8
⚡ High-Speed Processing — Processes video streams at 25–30 FPS using TensorRT optimization
🚑 Adaptive Signal Control — Dynamically adjusts green-light durations based on live traffic density
📊 Congestion Analysis — Analyzes vehicle density per zone to prioritize signal allocation
🛣️ Scalable Architecture — Designed to handle multiple intersections simultaneously


🛠️ Tech Stack

CategoryToolsLanguagePythonObject DetectionYOLOv8Computer VisionOpenCVInference OptimizationTensorRTImage ProcessingNumPy, PIL


📁 Project Structure

Traffic-management-system/
│
├── traffic-management-system/
│   ├── detection/         # Vehicle detection modules
│   ├── signal_control/    # Adaptive signal logic
│   ├── utils/             # Helper functions
│   └── main.py            # Entry point
│
└── README.md


🚀 Getting Started

Prerequisites

bashPython 3.8+
OpenCV
Ultralytics YOLOv8
TensorRT (optional, for GPU acceleration)

Installation

bash# Clone the repository
git clone https://github.com/siya-25/Traffic-management-system.git
cd Traffic-management-system

# Install dependencies
pip install ultralytics opencv-python numpy

Run the System

bashcd traffic-management-system
python main.py


📊 Results

MetricValueProcessing Speed25–30 FPSTraffic Zones Monitored4Throughput Improvement35%+Vehicle Instances Processed10,000+


💡 How It Works


Video Input — Camera feed from 4 traffic zones is captured in real time
Vehicle Detection — YOLOv8 detects and classifies all vehicles in each frame
Density Analysis — Vehicle count per zone is calculated to determine congestion level
Signal Optimization — Green-light duration is dynamically adjusted based on density
Continuous Loop — The system repeats every signal cycle for adaptive control


🔮 Future Improvements

 Emergency vehicle priority detection
 Multi-intersection coordination
 Web dashboard for real-time monitoring
 Integration with city traffic management APIs


📄 License

This project is open source and available under the MIT License.
