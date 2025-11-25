#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import yaml
import blobconverter
from pydobot import Dobot
from serial.tools import list_ports
from collections import deque
import serial.tools.list_ports

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

# YOLO Model Config
# - - - - - - - - - - - - - - -
model_name = "yolov5n_coco_416x416"
zoo_type = "depthai"

# Calibration File
CALIB_FILE = "camera_robot_calibration.yaml"

# Exponential smoothing factor for filtered 3D target position
# P_smooth = ALPHA * P_new + (1 - ALPHA) * P_old
ALPHA = 0.3

# Workspace / safety limits (robot frame, meters)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
REACH_MIN = 0.15 # min radius from base (avoid EE getting too close)
REACH_MAX = 0.30 # max reach (Arm has ~0.32 cm reach)
Z_MIN = 0.01 # 5 cm above table
Z_MAX = 0.175 # 17.5 cm above table

# Target YOLO classes (COCO names) and key bindings
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
TARGET_CLASSES = ["banana", "bottle", "person"] # 3 classes
# Swaps which object is being targeted based on user key press
TARGET_KEYS = ["a", "b", "c"] # 'a' = 0, 'b' = 1, 'c' = 2

# Confidence threshold
CONF_THRESH = 0.45

# Timeout for lost target (seconds)
TARGET_LOST_TIMEOUT = 3.0

CONTROL_PERIOD = 0.2  # seconds between robot commands (~5 Hz), tune as you like

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
# Robot Interface (using pydobot)
# ---------------------------------------------------------------------------
# x_mm, y_mm, z_mm = Cartesian position in millimeters
# r_deg = rotation of end effector in degrees
# isQueued=1 = add command to Dobot’s internal command queue
# PTP Type, PTPMOVLXYZ = Linear in Cartesian XYZ (straight line)

class RobotInterface:
    
    # Dobot Initialization
    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    def __init__(self):
        # Connect to Dobot Magician ("" = auto-select first arm found, USB baud rate)
        ports = [port.device for port in serial.tools.list_ports.comports()
         if 'ttyUSB' in port.device or 'ttyACM' in port.device]

        if len(ports) == 0:
            raise RuntimeError("No serial ports found for Dobot.")
        port = ports[0]
        print(f"Using port: {port}")

        # pick first port automatically (same behavior as previous code auto-connect)
        self.device = Dobot(port=port, verbose=False)

        # Set speed & initialize rate limiting
        self.device.speed(velocity=150, acceleration=150)
        
        curr_pos = self.device.pose()
        print(f"Connected to Dobot on {port}")
        print(f"Current Position: {curr_pos}")
      
        # Define Robot Sleep Pose
        self.SLEEP_X = 0.160
        self.SLEEP_Y = 0.00
        self.SLEEP_Z = 0.130
        self.SLEEP_R = 0.0

        # Store last commanded pose (x, y, z)
        self.last_command = None
        
        # Rate-limiting parameters for point_at()
        self.last_send_time = 0.0   # time.time() of last command
        self.min_send_dt = 0.0      # minimum time between commands (s)
        self.min_move_m = 0.01     # minimum movement (m) ≈ 5 mm

    # Sleep Function
    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # x_mm, y_mm, z_mm = Cartesian position in millimeters
    # r_deg = rotation of end effector in degrees
  
    def go_to_sleep(self):
        print("[ROBOT] Going to sleep pose...")
        self.device.move_to(200.0,
                            0.0,
                            100.0,
                            0.0, 
                            wait=False)

    # Object Tracking Movement
    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Command end-effector to point at (x,y,z) in robot base frame (meters).
    # Orientation kept fixed (pointing down) 
  
    def point_at(self, x_m, y_m, z_m):
        now = time.time()

        if self.last_command is not None:
            lx, ly, lz = self.last_command
            dist = ((x_m - lx)**2 + (y_m - ly)**2 + (z_m - lz)**2) ** 0.5
            if dist < self.min_move_m:
                # Too small movement, skip
                return

        x_mm = x_m * 1000.0
        y_mm = y_m * 1000.0
        z_mm = z_m * 1000.0

        print(f"[ROBOT] meters: ({x_m:.3f}, {y_m:.3f}, {z_m:.3f}) -> "
              f"mm: ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f})")

        self.last_command = (x_m, y_m, z_m)
        self.device.move_to(x_mm, y_mm, z_mm, self.SLEEP_R, wait=False)

    # Emergency Stop
    # - - - - - - - - - - - - - - - - - - - - - - - - - -    
    def emergency_stop(self):
        print("EMERGENCY STOP!")
        # pydobot does not support queue stop; no-op
        # Optionally turn off pump or motors
        # self.device.suck(False)
        pass

# ---------------------------------------------------------------------------
# Load calibration: R, t (camera -> robot)
# ---------------------------------------------------------------------------
# Open camera_robot_calibration.yaml and parses it into a Python dictionary
def load_calibration(calib_file=CALIB_FILE):
    with open(calib_file, "r") as f:
        data = yaml.safe_load(f)

    # Get rotation matrix & translation vector (camera wrt robot frame)
    R = np.array(data["rotation_matrix"], dtype=float)
    t = np.array(data["translation_m"], dtype=float).reshape(3)

    # Display to user
    print("Calibration: Loaded camera -> robot transform:")
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
    yolo.setCoordinateSize(4) # 4 coordinates per bounding box

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
    yolo.input.setBlocking(False) # Allow parallel inference
    yolo.input.setQueueSize(1) # Minimize queue size, avoid stale frame build-up

    return yolo

# ----------------------------------------------------
# Create Pipeline
# ----------------------------------------------------
def create_pipeline():
    pipeline = dai.Pipeline()

    # RGB camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(416, 416)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    # Mono cameras for stereo depth
    mono_left = pipeline.createMonoCamera()
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    mono_right = pipeline.createMonoCamera()
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Stereo Depth
    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
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
#    depth_frame_mm: 2D array from OAK (mm).
#    (cx, cy): center pixel of the detection (in depth frame coordinates).
#    roi_size: size of square region around that pixel.
#     min_valid_mm/max_valid_mm: clamp valid depth range.

def measure_distance(depth_frame_mm, cx, cy, roi_size=9,
                     min_valid_mm=300, max_valid_mm=15000):
                         
    h, w = depth_frame_mm.shape[:2] # Get height & width

    # Get bounds of square window centered at (cx, cy), clamp to image boundaries
    half = roi_size // 2
    x1 = max(0, cx - half)
    x2 = min(w - 1, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h - 1, cy + half)

    # Extract RoI
    roi = depth_frame_mm[y1:y2+1, x1:x2+1].astype(np.float32)

    # Keep only valid depths within range
    roi = roi[(roi > 0) & (roi >= min_valid_mm) & (roi <= max_valid_mm)]

    # Filter out zero/invalid values
    if roi.size == 0:
        return None
        
    # Return median depth in m (mm -> m)
    return float(np.median(roi)) / 1000.0

# ---------------------------------------------------------------------------
# Apply R, t: Convert Camera Frame -> Robot Frame
# ---------------------------------------------------------------------------
def camera_to_robot(P_cam, R, t):
    # Multiply 3D point in camera frame by R & add t
    # P_cam = np.array([x, y, z])
    return R @ P_cam + t

# ---------------------------------------------------------------------------
# Workspace limiting (reach + z clamp)
# ---------------------------------------------------------------------------
def clamp_workspace(P_robot):
    x, y, z = P_robot # Split 3D point into components (x, y, z)
    r = np.sqrt(x**2 + y**2) # radial distance from base (XY plane)

    # Scale outward if too close to base
    if r < REACH_MIN and r > 1e-4:
        scale = REACH_MIN / r
        x *= scale
        y *= scale
        print("Workspace Below REACH_MIN, scaling outward")

    # Scale inward if too far
    if r > REACH_MAX:
        scale = REACH_MAX / r
        x *= scale
        y *= scale
        print("Workspace Above REACH_MAX, scaling inward")

    # Clamp Z
    z_clamped = max(Z_MIN, min(Z_MAX, z))
    if z_clamped != z:
        print("Workspace: Clamping Z from", z, "to", z_clamped)
    z = z_clamped

    # Return a safe robot-frame position
    return np.array([x, y, z], dtype=float)

# ---------------------------------------------------------------------------
# Main visual servoing loop
# ---------------------------------------------------------------------------
def main():
    # Load calibration
    R, t = load_calibration()

    # Initialize robot interface
    robot = RobotInterface()
    robot.go_to_sleep()

    # DepthAI pipeline
    pipeline = create_pipeline()

    # State variables
    tracking_enabled = False # Whether robot is actively tracking
    current_target_index = 0  # 0,1,2 -> TARGET_CLASSES
    current_target_class = TARGET_CLASSES[current_target_index] # Which object is being tracked

    last_target_time = 0.0 # Last time there was valid detection
    P_smooth = None # Smoothed 3D target position

    # FPS tracking
    frame_count = 0
    t0 = time.time()

    last_control_time = time.time()

    # Resizable window for visualization
    cv2.namedWindow("visual_servo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("visual_servo", 900, 600)

    # Load pipeline into OAK Camera
    with dai.Device(pipeline) as device:
        # Queue size = 4, will store 4 frames
        # maxSize=4, blocking=False avoids app stalling if one stream lags; old frames drop instead
        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue("detections", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        print("Info: Visual servoing started.")
        print("      't' - toggle tracking on/off")
        print("      'a'/'b'/'c' - switch target class")
        print("      'q' - emergency stop & quit")

        # --------------------------------
        # Main Loop
        # --------------------------------
        while True:
            inRgb = qRgb.get()
            inDet = qDet.get()
            inDepth = qDepth.get()

            frame = inRgb.getCvFrame() # Convert RGB -> OpenCV BGR
            depth_frame = inDepth.getFrame() # Raw depth, 2D array (mm)

            # Get height/width of RGB & depth images
            h_rgb, w_rgb = frame.shape[:2]
            h_depth, w_depth = depth_frame.shape[:2]

            # Update frame count & compute FPS
            frame_count += 1
            fps = frame_count / (time.time() - t0)

            detections = inDet.detections # Get list of YOLO detection objects

            # Choose best detection of current target class
            best_det = None
            best_conf = 0.0

            for det in detections:
                if det.label >= len(LABEL_MAP): # Skip invalid labels
                    continue
                label = LABEL_MAP[det.label] # Convert # label -> string
                conf = det.confidence

                # Color for visualization
                if label == current_target_class:
                    color = (0, 255, 0)  # bright green for target class
                else:
                    color = (100, 100, 100)  # gray for others

                # Bounding box in RGB pixels
                # Convert normalized YOLO coords (0–1) to pixel coords
                # Clamp to image boundaries
                x1 = int(det.xmin * w_rgb)
                y1 = int(det.ymin * h_rgb)
                x2 = int(det.xmax * w_rgb)
                y2 = int(det.ymax * h_rgb)
                x1 = max(0, min(x1, w_rgb - 1))
                x2 = max(0, min(x2, w_rgb - 1))
                y1 = max(0, min(y1, h_rgb - 1))
                y2 = max(0, min(y2, h_rgb - 1))

                # Draw bounding box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cx_rgb = (x1 + x2) // 2
                cy_rgb = (y1 + y2) // 2

                # Get center of bounding box in pixel coords & draw cross
                cv2.drawMarker(frame, (cx_rgb, cy_rgb),
                               color, markerType=cv2.MARKER_CROSS,
                               markerSize=8, thickness=2)

                # Class & Confidence Labels
                label_txt = f"{label} {conf*100:.1f}%"
                cv2.putText(frame, label_txt,
                            (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Track only current target class w/ confidence threshold
                if label == current_target_class and conf > CONF_THRESH:
                    if conf > best_conf:
                        best_conf = conf
                        best_det = (cx_rgb, cy_rgb, conf)

            # Initialize variables for debugging/overlay
                        # Initialize variables for debugging/overlay
            P_robot_target = None
            depth_str = "N/A"

            # Unpack best detection center & confidence
            if best_det is not None:
                cx_rgb, cy_rgb, conf = best_det

                # Map RGB center to depth resolution
                cx_depth = int(cx_rgb * w_depth / w_rgb)
                cy_depth = int(cy_rgb * h_depth / h_rgb)

                # Depth in meters (using RoI)
                depth_m = measure_distance(depth_frame, cx_depth, cy_depth, roi_size=9)

                valid_3d = False
                if depth_m is None:
                    depth_str = "N/A"
                else:
                    depth_str = f"{depth_m:.2f} m"
                    # Use reasonable depth band; adjust if needed
                    if DEPTH_MIN_M <= depth_m <= DEPTH_MAX_M:
                        valid_3d = True
                    else:
                        depth_str = f"{depth_m:.2f} m (out of range)"

                if valid_3d:
                    # Pinhole model: camera frame coordinates
                    X_cam = (cx_rgb - CX) * depth_m / FX
                    Y_cam = (cy_rgb - CY) * depth_m / FY
                    Z_cam = depth_m

                    # Build 3D vector in camera frame
                    P_cam = np.array([X_cam, Y_cam, Z_cam], dtype=float)

                    # Camera -> robot frame
                    P_robot = camera_to_robot(P_cam, R, t)

                    # Optional sanity check on Z to avoid crazy values
                    if -0.05 < P_robot[2] < 0.40:
                        # Workspace Limiting
                        P_robot = clamp_workspace(P_robot)
                        P_robot_target = P_robot

                        # Exponential Smoothing - Reduce Jitter
                        if P_smooth is None:
                            P_smooth = P_robot
                        else:
                            P_smooth = ALPHA * P_robot + (1.0 - ALPHA) * P_smooth
                        
                        # Update time when target was last seen
                        last_target_time = time.time()

                        # If tracking enabled, send smoothed position to robot
                        if tracking_enabled and P_smooth is not None:
                            now = time.time()
                            if now - last_control_time >= CONTROL_PERIOD:
                                last_control_time = now
                                robot.point_at(P_smooth[0], P_smooth[1], P_smooth[2])


                        # Show 3D Robot position on frame
                        pos_text = (
                            f"Robot XYZ: ({P_smooth[0]:.3f}, "
                            f"{P_smooth[1]:.3f}, {P_smooth[2]:.3f}) m"
                        )
                        cv2.putText(
                            frame, pos_text, (10, h_rgb - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                        )
                    else:
                        print(f"[WARN] Rejecting P_robot with bad Z: {P_robot}")

            # If no target for a while and tracking is on -> go to sleep
            now = time.time()
            if tracking_enabled and (now - last_target_time > TARGET_LOST_TIMEOUT):
                print("Target lost, returning to sleep pose.")
                tracking_enabled = False
                robot.go_to_sleep()
                P_smooth = None

            # Status overlays
            # - - - - - - - - - - - - - - - - - - - - - -
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
            
            # Display annotated frame
            cv2.imshow("visual_servo", frame)

            # 'q' = emergency stop -> send robot to sleep & exit program
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Emergency stop & exit.")
                robot.emergency_stop()
                robot.go_to_sleep()
                break
                
            # 't' = toggle tracking ON/OFF. OFF -> robot to sleep
            elif key == ord('t'):
                tracking_enabled = not tracking_enabled
                print(f"[INFO] Tracking {'ENABLED' if tracking_enabled else 'DISABLED'}")
                if not tracking_enabled:
                    robot.go_to_sleep()
                    
            # 'a', 'b', 'c' = switch which object class to track
            elif key in [ord('a'), ord('b'), ord('c')]:
                idx = TARGET_KEYS.index(chr(key))
                current_target_index = idx
                current_target_class = TARGET_CLASSES[current_target_index]
                print(f"[INFO] Switched target class to: {current_target_class}")
                P_smooth = None  # reset smoothing

        # Close OpenCV windows after breaking from loop
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

