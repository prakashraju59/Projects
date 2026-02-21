## 📌 Project Goal

This project builds a **camera-based Lane Detection and Lane Departure Warning (LDW)** system using classical image processing.

The system detects lane markings from a monocular camera and gives a warning when the vehicle drifts near a lane boundary.

This is basic LDW functions used in ADAS systems.

---

## 🛠️ Tools & Software Used

* Python
* OpenCV
* ROS 2
* NumPy
* Image Processing Techniques

---

## ⚙️ System Pipeline

1. Camera calibration and image undistortion
2. Perspective transform → Bird’s-eye view
3. HLS color + Sobel gradient filtering
4. Sliding window lane detection
5. Polynomial lane fitting
6. Solid / dashed lane classification
7. Lane Departure Warning logic

---

## 📊 Key Features

✔ Detects left and right lane boundaries
✔ Classifies lanes as **solid or dashed**
✔ Calculates distance from vehicle to lane edge
✔ Gives warning if vehicle drifts near line
✔ Runs in real-time using ROS 2

---

## 📉 Limitations

* Needs clear lane markings
* Sensitive to shadows and night driving
* Assumes flat road
* Not using vehicle dynamics yet
* low speed maneuvering

---

## 🚀 Future Improvements

* Add IMU for slope correction
* Use deep learning lane detection
* Add Lane Keeping Assist control
* Improve night and rain performance

---

## ▶️ How to Run

1. Install Python + OpenCV + ROS2
2. Run the ROS2 node:

