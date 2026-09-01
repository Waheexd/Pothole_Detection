# 🛣️ Road Pothole Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://potholedetectorsystem.streamlit.app)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![YOLO](https://img.shields.io/badge/model-YOLOv8-orange.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

An AI-powered computer vision application designed to inspect, detect, and evaluate road surface defects and potholes in real-time. Built for deployment in **connected vehicle dashcams**, **ADAS perception pipelines**, **autonomous vehicle navigation**, and **municipal road audit tools**, powered by a custom-trained **Ultralytics YOLOv8** model and a modern, theme-adaptive glassmorphic dashboard.

🔗 **Live Deployment:** [potholedetectorsystem.streamlit.app](https://potholedetectorsystem.streamlit.app)

---

## 📸 Workflow & Working Explanation

The application provides an end-to-end visual inspection pipeline from image ingestion to detailed defect analytics:

### 1. Ingestion & Instant Detection Preview
Upload road surface imagery in JPG, JPEG, or PNG format. The dashboard loads the custom YOLO weights (`best.pt`) once into memory using `@st.cache_resource` and offers real-time sensitivity controls via the sidebar.

Upon clicking **🔎 Detect Potholes**, the image preview dynamically switches to reveal interactive detection tabs without requiring scrolling:

<p align="center">
  <img src="assets/detection_preview.png" alt="Detection Preview" width="900">
</p>

- **Interactive Tab Toggle:** Easily flip between the **🎯 Detection Result** (annotated with localized bounding boxes and confidence tags like `pothole 0.85`, `pothole 0.73`) and the **🖼️ Original Image**.
- **Real-Time Status Confirmation:** Immediate visual feedback alerts the user to the exact number of identified hazards.
- **Sidebar Control Panel:** Fine-tune the **Confidence Threshold** slider (10% to 90%) to control detector sensitivity for different road textures and lighting conditions.

---

### 2. Side-by-Side Comparative Analysis & Session Metrics
Directly beneath the preview, the dashboard generates a full comparative breakdown comparing the raw road surface against the model's bounding-box annotations, accompanied by executive summary metric cards:

<p align="center">
  <img src="assets/detection_results.png" alt="Detection Results & Metrics" width="900">
</p>

- **Side-by-Side Image Comparison:** Clearly contrasts original road conditions against annotated defect regions.
- **Total Potholes Detected:** Quantifies total hazards identified across the surface area.
- **Highest Confidence Score:** Highlights the peak certainty score among all detections (e.g., `85.00%`).
- **Inference Speed:** Benchmarks YOLO execution latency in milliseconds (`~96 ms`), verifying production-level real-time performance.

---

### 3. Detailed Detection Breakdown & Reliability Tiers
Every detected road hazard is cataloged in a structured breakdown table that classifies defects according to confidence reliability tiers:

<p align="center">
  <img src="assets/confidence_breakdown.png" alt="Confidence Breakdown Table" width="900">
</p>

- **Defect Indexing:** Each pothole candidate is assigned an individual tracking index (`Pothole #1`, `Pothole #2`, etc.) sorted in descending order of certainty.
- **Confidence Rating:** Exact percentage certainty reported by the neural network.
- **Reliability Classification:**
  - 🟢 **High Confidence (≥ 75%):** Well-defined, critical surface hazards requiring immediate road maintenance attention.
  - 🟡 **Medium Confidence (45% - 74%):** Developing surface wear, early-stage potholes, or partially obscured depressions.
  - 🔴 **Low Confidence (< 45%):** Minor road anomalies, surface shadows, or candidates near threshold limits.

---

## 🧠 System Architecture & Pipeline

```mermaid
graph LR
    A[📷 Input Surface Image] --> B[🔄 Preprocessing & Orientation]
    B --> C[⚡ YOLOv8 Deep Learning Inference]
    C --> D[📦 Bounding Box & Confidence Extraction]
    D --> E[🎯 Instant Result Tabs Preview]
    D --> F[📊 Side-by-Side Comparison]
    D --> G[📈 Session Metrics & Speed Benchmark]
    D --> H[📋 Reliability Tier Breakdown Table]
```

---

## ✨ Core Features

- **🚗 In-Vehicle Real-Time ADAS & Connected Car Readiness:**
  - Designed for real-time integration into **dashcams**, **Advanced Driver Assistance Systems (ADAS)**, and **autonomous vehicles**.
  - High-speed inference (~15–30 ms on GPU, ~90 ms on CPU) delivers instantaneous hazard alerts at driving speeds, giving human drivers and collision-avoidance systems critical reaction time to steer around deep potholes or brake safely.
- **🧠 Custom Fine-Tuned YOLOv8 Architecture:**
  - Specialized single-stage neural network weights (`best.pt`, ~5.4MB) trained specifically to isolate road cracks, sunken asphalt, water-filled potholes, and surface depressions across diverse lighting and weather conditions.
- **⚡ Lightweight Edge & Embedded Hardware Deployment:**
  - Extremely lightweight footprint makes it ready to deploy on in-car edge computing hardware such as **NVIDIA Jetson Nano / Orin**, **Raspberry Pi 5 with AI accelerators**, or vehicle telemetry units without requiring heavy cloud connections.
- **📊 Quantitative Defect Risk Scoring & Reliability Tiers:**
  - Classifies detections into categorized reliability tiers (**High ≥ 75%**, **Medium 45–74%**, **Low < 45%**) to separate immediate collision risks from minor road surface wear.
- **🏙️ Smart City Fleet & Municipal Telematics Integration:**
  - Ideal for mounting on municipal buses, delivery fleets, or utility vehicles to automatically scan and log road network quality, enabling automated GIS mapping and prioritized maintenance dispatch.
- **🎨 Responsive Glassmorphic Dashboard with Live Analytics:**
  - Premium theme-aware UI supporting instant in-place tab previewing, real-time threshold adjustments, latency benchmarking, and persistent session state (`st.session_state`) in both **Light** and **Dark** modes.

---

## 🚘 Real-World Automotive & Smart Mobility Use Cases

| Domain | Application | Benefit |
| :--- | :--- | :--- |
| **ADAS & Smart Dashcams** | Forward-facing vehicle cameras scanning the driving path | Gives audible/visual alerts to drivers or triggers active suspension to soften impact |
| **Autonomous Vehicles (AVs)** | Perception layer sensor fusion alongside LiDAR and RADAR | Informs path-planning algorithms to safely maneuver around road craters |
| **Municipal Road Audit Fleets** | Public transit buses and garbage trucks equipped with edge cameras | Automatically flags damaged road coordinates to city public works departments |
| **Fleet & Logistics Management** | Commercial delivery vans and freight trucks | Prevents wheel, suspension, and tire damage, reducing maintenance downtime |

---

## 📁 Repository Structure

```text
Pothole_Detection/
├── assets/
│   ├── detection_preview.png       # UI upload & real-time detection tab screenshot
│   ├── detection_results.png       # Side-by-side comparison and summary metrics
│   └── confidence_breakdown.png    # Detailed detection breakdown table
├── .gitignore                      # Git exclusion rules
├── app.py                          # Streamlit application with YOLOv8 pipeline & custom UI
├── best.pt                         # Custom-trained YOLO model weights
├── requirements.txt                # Python package dependencies
└── README.md                       # Documentation (this file)
```

---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Waheexd/Pothole_Detection.git
cd Pothole_Detection
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
*(Or directly via virtual environment executable: `.\.venv\Scripts\streamlit.exe run app.py`)*

Open your browser and navigate to **`http://localhost:8501`**.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Fork or push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and click **Deploy an app**.
3. Select your repository (`Waheexd/Pothole_Detection`), set branch to `master`, and main file path to `app.py`.
4. Click **Deploy**!
