# Object Tracking & Visual Servoing (OAK-D + Dobot)

This module implements **real-time 3D object tracking and visual servoing** using:

- A **Luxonis OAK-D** camera (RGB + stereo depth)
- A **Dobot** robotic arm (via `pydobot`)
- **YOLOv5n** object detection running on the OAK-D Myriad X VPU
- A pre-computed **camera-to-robot calibration** (`camera_robot_calibration.yaml`)

The robot continuously tracks a selected object class (e.g., banana, bottle, person) in 3D and moves its end-effector to follow the target inside a safe workspace.

---

## Script

### `object_tracking.py`

Main features:

- Streams RGB (416×416) + stereo depth from the OAK-D
- Runs YOLOv5n (`yolov5n_coco_416x416`) on-device via DepthAI + blobconverter
- Estimates 3D target position in **camera frame** using:
  - Depth ROI median (9×9)
  - Camera intrinsics (`FX`, `FY`, `CX`, `CY`)
- Transforms 3D points into **robot base frame** using calibration:
  - `P_robot = R @ P_cam + t`
  - Calibration loaded from `camera_robot_calibration.yaml`
- Applies workspace limits and smoothing:
  - Radial reach constraints (`REACH_MIN`, `REACH_MAX`)
  - Z height clamp (`Z_MIN`, `Z_MAX`)
  - Exponential smoothing of 3D target (`ALPHA`)
- Sends non-blocking target commands to **Dobot** via a background control thread
- Logs performance metrics:
  - Tracking error (Euclidean error between commanded and actual EE pose)
  - Success rate (frames with valid 3D target while tracking is enabled)
  - Workspace coverage (min/max XYZ of commanded poses)
  - FPS and frame counts

---

## Target Classes & Controls

The script focuses on a small subset of COCO classes:

```python
TARGET_CLASSES = ["banana", "bottle", "person"]
TARGET_KEYS    = ["a", "b", "c"]  # 'a' → banana, 'b' → bottle, 'c' → person
