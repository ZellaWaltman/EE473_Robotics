# Hand–Eye Calibration (OAK-D + Dobot + AprilTag)

This module performs **hand–eye calibration** between:

- The **Dobot end-effector frame** (EE)
- An **AprilTag** mounted on the end-effector
- A **Luxonis OAK-D** camera, previously calibrated to the robot base

It then runs **real-time end-effector tracking**, comparing:

- EE pose from **forward kinematics (FK)** (via Dobot encoders)
- EE pose from **vision** (AprilTag detections + hand–eye + camera–robot calibration)

---

## Main Script

### `handeye_calibration.py`

This script has four main phases:

1. **Data Collection (Hand–Eye Samples)**
2. **Hand–Eye Calibration (Solving AX = XB)**
3. **Error Analysis & Quality Check**
4. **Real-Time End-Effector Tracking & Visualization**

---

## Prerequisites

### Hardware

- **Luxonis OAK-D** camera connected via USB
- **Dobot** robotic arm (e.g., Magician) connected via USB
- An **AprilTag (ID 4, tag36h11 family)** rigidly mounted on the **end-effector**
  - Tag physical size: **38 mm × 38 mm**

### Required Calibration File

You must already have a **camera-to-robot-base calibration** from an earlier step (e.g., AprilTag calibration):

- `camera_robot_calibration.yaml`

It should contain:

- `rotation_matrix` (3×3, camera → robot base)
- `translation_m` (3×1, meters)

Example structure:

```yaml
rotation_matrix:
  - [r11, r12, r13]
  - [r21, r22, r23]
  - [r31, r32, r33]
translation_m: [tx, ty, tz]
