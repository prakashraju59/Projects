# Automotive Active Safety & ADAS Projects

This repository contains technical reports and implementations from the **Active Safety (TME192)** and **Automotive Engineering Project (TME180)** courses at Chalmers University of Technology.

## 📂 Project Structure

### 1. ADAS Prototyping Platform (ROS 2) [Report](http://hdl.handle.net/20.500.12380/310975)
* **Objective:** Build a small-scale Driver-in-the-Loop testing platform.
* **Key Contributions:** * Designed a ROS 2 architecture for sensor fusion (LiDAR + Camera).
    * Implemented **AEB logic** based on stopping-distance triggers.
    * Set up a UDP-based head-tracking system for VR-controlled pan-tilt cameras.

### 2. Computer Vision: Lane Detection
* **Objective:** Robust lane marking detection for autonomous lateral control.
* **Technical Highlights:**
    * **Preprocessing:** Camera calibration and image undistortion.
    * **Filtering:** HLS color space masking combined with Sobel operators.
    * **Tracking:** Sliding window search and polynomial fitting to estimate lane curvature.

### 3. Safety Metric Analysis
* **Objective:** Analyzing vehicle-to-vehicle conflict scenarios.
* **Key Metrics:**
    * **Time-to-Collision (TTC):** Calculated limits for FCW and AEB activation.
    * **Braking Logic:** Defined $a_{req}$ (required deceleration) vs $a_{max}$ thresholds.
    * **Signal Processing:** Used Savitzky-Golay filters to extract meaningful trends from noisy radar data.

## 🛠️ Tools Used
* **Software:** Python 3.x, ROS 2 Humble, MATLAB.
* **Libraries:** OpenCV, NumPy, Matplotlib.
* **Hardware:** MentorPi Robot, Meta Quest 3, 360° LiDAR.

## 📄 Documentation
Detailed technical reports for each project are available in the `/reports` folder. 
*(Note: Personal contact information of teammates has been redacted for privacy.)*
