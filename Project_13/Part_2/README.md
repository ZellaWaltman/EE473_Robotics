# AprilTag Camera–Robot Calibration (OAK-D + AprilTags)

This section contains utilities for calibrating an OAK-D camera to a robot base frame using AprilTags, and for visualizing / validating the calibration.

It assumes:

- **Camera:** Luxonis OAK-D (color camera used at 416×416 preview)
- **Tags:** AprilTag family `tag36h11`, size **38 mm** (0.038 m)
- **Robot frame:** You already know the 3D positions of a few tags in the robot base frame and store them in `apriltag_known_positions.yaml`.

---

# 📂 Contents

apriltag_detector.py
apriltag_calibration.py
calibration_auto_check.py
apriltag_known_positions.yaml
camera_robot_calibration.yaml (generated after calibration)
requirements.txt
README.md

yaml
Copy code

---

# 🧭 Overview

This system solves for the rigid-body transform:

\[
P_{\text{robot}} = R \cdot P_{\text{camera}} + t
\]

Where:

- `R` — rotation (camera → robot base)
- `t` — translation (camera origin in robot base frame)
- AprilTags provide 3D positions in the **camera frame**
- Known tag positions in the robot base frame provide correspondence points

The calibration is performed using **point-cloud alignment (SVD-based rigid transform)**, giving an accurate mapping from OAK-D coordinates → robot base coordinates.

---

# Files & Their Purpose

## `apriltag_detector.py` — Live AprilTag Viewer

Use this script to verify:

- AprilTag detection quality  
- Pose estimation stability  
- Tag IDs & labeling  

It displays for each tag:

- 2D outline  
- 3D pose estimate (x,y,z in mm)  
- Roll / pitch / yaw  
- Decision margin  

**Run:**

```bash
python3 apriltag_detector.py
Press q to quit.

##**apriltag_known_positions.yaml — Ground-Truth Tag Positions**

This file defines the robot-frame coordinates of AprilTags you arranged for the calibration process.

Example:

tag_positions:
  0: [0.30,  0.20, 0.00]
  1: [0.30, -0.20, 0.00]
  2: [0.098, 0.20, 0.00]
  3: [0.098,-0.20, 0.00]

tag_size_m: 0.038

These positions must match your real-world layout.

##**apriltag_calibration.py — Computes Camera→Robot Calibration**

This script:

- Detects AprilTags {0, 1, 2, 3}
- Collects SAMPLES_PER_TAG 3D pose samples in the camera frame
- Averages camera-frame observations
- Solves for rotation R and translation t using SVD
- Computes per-tag error (mm)
- Saves output to: camera_robot_calibration.yaml

##**calibration_auto_check.py**

This script loads:
- The known true tag positions
- The calibration file produced earlier

For each visible AprilTag:
- Computes estimated robot-frame position
- Compares to the ground truth value
- Computes error in millimeters
- Colors each tag by accuracy:
    - Green = < 10 mm
    - Yellow = 10–20 mm
    - Red = > 20 mm

Displays summary in bottom right:
- FPS
- Mean error
- Max error
