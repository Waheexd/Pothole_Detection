# 🛣️ Road Pothole Detection System

An AI-powered computer vision dashboard designed to inspect and identify road potholes in uploaded imagery using a custom-trained **Ultralytics YOLO** model. The interface is built on **Streamlit** with a theme-aware premium layout, support for dark/light modes, performance metrics, and detailed confidence breakdown tables.

---

## 🚀 Features

- **🧠 YOLO Neural Network Integration**: Runs inference in real-time on uploaded road surface imagery.
- **⚡ High-Performance Caching**: Uses Streamlit resource caching (`@st.cache_resource`) to load model weights once, keeping predictions fast and saving system memory.
- **🎨 Premium Visual Dashboard**:
  - Custom glassmorphism headers and fonts (**Space Grotesk** & **Plus Jakarta Sans**).
  - Responsive, styled file upload dropzone with hover micro-animations.
  - Side-by-side original image vs. bounding-box annotated prediction outputs.
- **📊 Real-Time Analytics**:
  - Displays total pothole counts.
  - Displays highest confidence score percentage.
  - Reports inference speed (in milliseconds) for the YOLO execution pipeline.
  - Builds an HTML-formatted, color-coded reliability tier table (**High**, **Medium**, and **Low** confidence badges).
- **⚙️ Dynamic Parameter Tuning**: Adjustable confidence threshold slider in the sidebar to change detector sensitivity on the fly.
- **🌗 Theme Compatibility**: Full compatibility with both Streamlit Light and Dark modes.

---

## 📁 Repository Structure

```text
Pothole_Detection/
├── .gitignore          # Ignores local environments, pycache, and logs
├── app.py              # Main dashboard script containing UI and YOLO pipeline
├── best.pt             # Pre-trained YOLO weights (user-provided)
├── requirements.txt    # Application dependencies list
└── README.md           # Documentation (this file)
```

---

## 🛠️ Installation & Setup

Follow these steps to run the application locally on your system:

### 1. Prerequisites
Ensure you have Python (version 3.8 to 3.11 is recommended) installed.

### 2. Set Up a Virtual Environment (Recommended)
Create and activate a virtual environment in the project directory:

```bash
# Create the environment
python -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate it (Windows Command Prompt)
.venv\Scripts\activate.bat

# Activate it (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Provide Model Weights
Place your trained YOLO weights in the project root directory and ensure they are named:
`best.pt`

---

## 💻 Running the Application

Launch the Streamlit web server:

```bash
streamlit run app.py
```

The application will start, and a browser window will automatically open at:
`http://localhost:8501`

---

## 📝 How to Use

1. **Upload an Image**: Drag & drop or select a JPG, JPEG, or PNG road surface image.
2. **Configure Threshold**: Adjust the **Confidence Threshold** slider in the sidebar. (A lower threshold detects more potential potholes; a higher threshold filters out weaker matches).
3. **Trigger Detection**: Click **🔎 Detect Potholes**.
4. **Analyze Results**: Review the side-by-side visual comparison, metric statistics (potholes count, max confidence, speed), and the detailed breakdown table at the bottom.
