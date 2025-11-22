#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import yaml
import blobconverter
from collections import deque

# -------------------------------------------------------
# Camera intrinsics for 416x416 resolution
# -------------------------------------------------------
# focal lengths
FX = 450.0
FY = 450.0
# center coords
CX = 208.0
CY = 208.0

# Depth validity range (meters)
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 1.8

# YOLO model (DepthAI blob)
MODEL_NAME = "yolov5n_coco_416x416"
ZOO_TYPE = "depthai"

# Calibration files
CALIB_FILE = "camera_robot_calibration.yaml"

# Exponential smoothing factor
ALPHA = 0.3

# Workspace / safety limits (robot frame, meters)
REACH_MIN = 0.15     # min radius from base (avoid singularity)
REACH_MAX = 0.45     # max reach
Z_MIN = 0.05         # 5 cm above table
Z_MAX = 0.40         # 40 cm above table

# Target YOLO classes (COCO names) and key bindings
TARGET_CLASSES = ["bottle", "cup", "person"]  # choose 3
TARGET_KEYS = ["a", "b", "c"] # 'a' = 0, 'b' = 1, 'c' = 2

# Confidence threshold
CONF_THRESH = 0.5

# Timeout for lost target (seconds)
TARGET_LOST_TIMEOUT = 3.0

# COCO labels (80 classes)
# ----------------------------------------------------
LABEL_MAP = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ---------------------------------------------------------------------------
# Robot interface abstraction
# ---------------------------------------------------------------------------
class RobotInterface:
    def __init__(self):
        # TODO: initialize your robot here (Dobot Magician or RX-150)
        # For example, if using Interbotix:
        # from interbotix_xs_modules.arm import InterbotixManipulatorXS
        # self.bot = InterbotixManipulatorXS("rx150", "arm", "gripper")
        self.last_command = None

    def go_to_sleep(self):
        print("[ROBOT] Going to sleep pose...")
        # Example (Interbotix):
        # self.bot.arm.go_to_sleep_pose(moving_time=5.0, accel_time=2.0)
        # Example (Dobot): call Dobot API to move to safe home pose
        pass

    def point_at(self, x, y, z):
        """
        Command end-effector to point at (x,y,z) in robot base frame.
        Keep orientation fixed (e.g., pointing down).
        """
        print(f"[ROBOT] Pointing at x={x:.3f}, y={y:.3f}, z={z:.3f}")
        self.last_command = (x, y, z)
        # Example (Interbotix-style):
        # self.bot.arm.set_ee_pose_components(
        #     x=x, y=y, z=z,
        #     roll=0.0, pitch=1.5, yaw=0.0,
        #     moving_time=5.0, accel_time=2.0
        # )
        # For Dobot: use whatever API you have to move Cartesian pose slowly
        pass

    def emergency_stop(self):
        print("[ROBOT] EMERGENCY STOP (implement as needed)")
        # For some robots, you can call a stop / disable command here.
        pass

# ---------------------------------------------------------------------------
# Load calibration: R, t (camera -> robot)
# ---------------------------------------------------------------------------
def load_calibration(calib_file=CALIB_FILE):
    with open(calib_file, "r") as f:
        data = yaml.safe_load(f)

    R = np.array(data["rotation_matrix"], dtype=float)        # 3x3
    t = np.array(data["translation_m"], dtype=float).reshape(3)  # 3,

    print("[CALIB] Loaded camera->robot transform:")
    print("R =\n", R)
    print("t =", t)
    return R, t

# ----------------------------------------------------
# YOLOv5n detection network - Create node
# ----------------------------------------------------
def create_yolo_node(
    pipeline,
    model_name: str = model_name,
    zoo_type: str = zoo_type,
    conf_thr: float = 0.45,
    iou_thr: float = 0.45
):
    yolo = pipeline.create(dai.node.YoloDetectionNetwork)

    blob_path = blobconverter.from_zoo(
        name=model_name,
        shaves=3,
        zoo_type=zoo_type
    )

    yolo.setBlobPath(blob_path)
    
    # Yolo specific parameters
    yolo.setConfidenceThreshold(conf_thr)
    yolo.setIouThreshold(iou_thr)
    yolo.setNumClasses(80)
    yolo.setCoordinateSize(4)

    # Anchors for YOLOv5n 416x416 blob
    yolo.setAnchors([
        10, 13, 16, 30, 33, 23,
        30, 61, 62, 45, 59, 119,
        116, 90, 156, 198, 373, 326
    ])

    # 416x416 blob produces three output grids:
    #	- 52x52
    #   - 26x26
    #   - 13x13
    yolo.setAnchorMasks({
        "side52": [0, 1, 2],
        "side26": [3, 4, 5],
        "side13": [6, 7, 8],
    })

    yolo.setNumInferenceThreads(2)
    yolo.input.setBlocking(False)
    yolo.input.setQueueSize(1)

    return yolo

# ----------------------------------------------------
# Create Pipeline
# ----------------------------------------------------
def create_pipeline():
    pipeline = dai.Pipeline()

    # Color camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(416, 416)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    # Mono cameras for stereo depth
    mono_left = pipeline.createMonoCamera()
    mono_right = pipeline.createMonoCamera()
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Stereo Depth
    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth to RGB FOV

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # YOLO Node
    yolo = create_yolo_node(pipeline)
    cam_rgb.preview.link(yolo.input)

    # Output Streams
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_det = pipeline.createXLinkOut()
    xout_det.setStreamName("detections")
    yolo.out.link(xout_det.input)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline

# ----------------------------------------------------
# Measure depth (9×9 ROI median)
# ----------------------------------------------------
def measure_distance(depth_frame_mm, cx, cy, roi_size=9,
                     min_valid_mm=300, max_valid_mm=15000):

    h, w = depth_frame_mm.shape[:2]

    half = roi_size // 2
    x1 = max(0, cx - half)
    x2 = min(w - 1, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h - 1, cy + half)

    roi = depth_frame_mm[y1:y2+1, x1:x2+1].astype(np.float32)

    # Keep only valid depths within range
    roi = roi[(roi > 0) & (roi >= min_valid_mm) & (roi <= max_valid_mm)]

    if roi.size == 0:
        return None

    return float(np.median(roi))

# ---------------------------------------------------------------------------
# Apply R, t: camera → robot
# ---------------------------------------------------------------------------
def camera_to_robot(P_cam, R, t):
    # P_cam: np.array([x, y, z])
    return R @ P_cam + t

# ---------------------------------------------------------------------------
# Workspace limiting (reach + z clamp)
# ---------------------------------------------------------------------------
def clamp_workspace(P_robot):
    x, y, z = P_robot
    r = np.sqrt(x**2 + y**2)

    # Scale outward if too close
    if r < REACH_MIN and r > 1e-4:
        scale = REACH_MIN / r
        x *= scale
        y *= scale
        print("[WORKSPACE] Below REACH_MIN, scaling outward")

    # Scale inward if too far
    if r > REACH_MAX:
        scale = REACH_MAX / r
        x *= scale
        y *= scale
        print("[WORKSPACE] Above REACH_MAX, scaling inward")

    # Clamp Z
    z_clamped = max(Z_MIN, min(Z_MAX, z))
    if z_clamped != z:
        print("[WORKSPACE] Clamping Z from", z, "to", z_clamped)
    z = z_clamped

    return np.array([x, y, z], dtype=float)

# ---------------------------------------------------------------------------
# Main visual servoing loop
# ---------------------------------------------------------------------------
def main():
    # Load calibration
    R, t = load_calibration()

    # Init robot interface
    robot = RobotInterface()
    robot.go_to_sleep()

    # DepthAI pipeline
    pipeline = create_pipeline()

    # State variables
    tracking_enabled = False
    current_target_index = 0  # 0,1,2 → TARGET_CLASSES
    current_target_class = TARGET_CLASSES[current_target_index]

    last_target_time = 0.0
    P_smooth = None

    # FPS tracking
    frame_count = 0
    t0 = time.time()

    cv2.namedWindow("visual_servo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("visual_servo", 900, 600)

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue("detections", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        print("[INFO] Visual servoing started.")
        print("      't' - toggle tracking on/off")
        print("      'a'/'b'/'c' - switch target class")
        print("      'q' - emergency stop & quit")

        while True:
            inRgb = qRgb.get()
            inDet = qDet.get()
            inDepth = qDepth.get()

            frame = inRgb.getCvFrame()
            depth_frame = inDepth.getFrame()  # uint16 mm

            h_rgb, w_rgb = frame.shape[:2]
            h_depth, w_depth = depth_frame.shape[:2]

            frame_count += 1
            fps = frame_count / (time.time() - t0)

            detections = inDet.detections

            # Choose best detection of current target class
            best_det = None
            best_conf = 0.0

            for det in detections:
                if det.label >= len(LABEL_MAP):
                    continue
                label = LABEL_MAP[det.label]
                conf = det.confidence

                # Color for visualization
                if label == current_target_class:
                    color = (0, 255, 0)  # bright green for target class
                else:
                    color = (100, 100, 100)  # gray for others

                # Bounding box in RGB pixels
                x1 = int(det.xmin * w_rgb)
                y1 = int(det.ymin * h_rgb)
                x2 = int(det.xmax * w_rgb)
                y2 = int(det.ymax * h_rgb)
                x1 = max(0, min(x1, w_rgb - 1))
                x2 = max(0, min(x2, w_rgb - 1))
                y1 = max(0, min(y1, h_rgb - 1))
                y2 = max(0, min(y2, h_rgb - 1))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cx_rgb = (x1 + x2) // 2
                cy_rgb = (y1 + y2) // 2

                cv2.drawMarker(frame, (cx_rgb, cy_rgb),
                               color, markerType=cv2.MARKER_CROSS,
                               markerSize=8, thickness=2)

                label_txt = f"{label} {conf*100:.1f}%"
                cv2.putText(frame, label_txt,
                            (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Track only current target class with confidence threshold
                if label == current_target_class and conf > CONF_THRESH:
                    if conf > best_conf:
                        best_conf = conf
                        best_det = (cx_rgb, cy_rgb, conf)

            P_robot_target = None
            depth_str = "N/A"

            if best_det is not None:
                cx_rgb, cy_rgb, conf = best_det

                # Map RGB center to depth resolution
                cx_depth = int(cx_rgb * w_depth / w_rgb)
                cy_depth = int(cy_rgb * h_depth / h_rgb)

                # Depth in meters
                depth_m = measure_distance(depth_frame, cx_depth, cy_depth, roi_size=9)

                if depth_m is not None:
                    depth_str = f"{depth_m:.2f} m"

                    # Pinhole model: camera frame coordinates
                    X_cam = (cx_rgb - CX) * depth_m / FX
                    Y_cam = (cy_rgb - CY) * depth_m / FY
                    Z_cam = depth_m

                    P_cam = np.array([X_cam, Y_cam, Z_cam], dtype=float)
                    P_robot = camera_to_robot(P_cam, R, t)

                    # Workspace limiting
                    P_robot = clamp_workspace(P_robot)
                    P_robot_target = P_robot

                    # Exponential smoothing
                    if P_smooth is None:
                        P_smooth = P_robot
                    else:
                        P_smooth = ALPHA * P_robot + (1.0 - ALPHA) * P_smooth

                    last_target_time = time.time()

                    # If tracking enabled, send command
                    if tracking_enabled:
                        robot.point_at(P_smooth[0], P_smooth[1], P_smooth[2])

                    # Show 3D pos on frame
                    pos_text = f"Robot XYZ: ({P_smooth[0]:.3f}, {P_smooth[1]:.3f}, {P_smooth[2]:.3f}) m"
                    cv2.putText(frame, pos_text, (10, h_rgb - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # If no target for a while and tracking is on -> go to sleep
            now = time.time()
            if tracking_enabled and (now - last_target_time > TARGET_LOST_TIMEOUT):
                print("[INFO] Target lost, returning to sleep pose.")
                tracking_enabled = False
                robot.go_to_sleep()
                P_smooth = None

            # Status overlay
            status_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, status_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            track_text = f"Tracking: {'ON' if tracking_enabled else 'OFF'}"
            cv2.putText(frame, track_text, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if tracking_enabled else (0, 0, 255), 1)

            target_text = f"Target: {current_target_class}"
            cv2.putText(frame, target_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            depth_text = f"Depth: {depth_str}"
            cv2.putText(frame, depth_text, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("visual_servo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Emergency stop & exit.")
                robot.emergency_stop()
                robot.go_to_sleep()
                break
            elif key == ord('t'):
                tracking_enabled = not tracking_enabled
                print(f"[INFO] Tracking {'ENABLED' if tracking_enabled else 'DISABLED'}")
                if not tracking_enabled:
                    robot.go_to_sleep()
            elif key in [ord('a'), ord('b'), ord('c')]:
                idx = TARGET_KEYS.index(chr(key))
                current_target_index = idx
                current_target_class = TARGET_CLASSES[current_target_index]
                print(f"[INFO] Switched target class to: {current_target_class}")
                P_smooth = None  # reset smoothing

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
