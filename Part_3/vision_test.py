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
FX = 450.0
FY = 450.0
CX = 208.0
CY = 208.0

DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 1.8

model_name = "yolov5n_coco_416x416"
zoo_type = "depthai"
CALIB_FILE = "camera_robot_calibration.yaml"
ALPHA = 0.3

REACH_MIN = 0.15
REACH_MAX = 0.30
Z_MIN = 0.05
Z_MAX = 0.175

TARGET_CLASSES = ["banana", "bottle", "person"]
TARGET_KEYS = ["a", "b", "c"]
CONF_THRESH = 0.45
TARGET_LOST_TIMEOUT = 3.0

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
class RobotInterface:
    def __init__(self):
        ports = [port.device for port in serial.tools.list_ports.comports()
         if 'ttyUSB' in port.device or 'ttyACM' in port.device]

        if len(ports) == 0:
            raise RuntimeError("No serial ports found for Dobot.")
        port = ports[0]
        print(f"Using port: {port}")

        # pick first port automatically (same behavior as previous code auto-connect)
        self.device = Dobot(port=port, verbose=False)
        
        curr_pos = self.device.pose()
        print(f"Connected to Dobot on {port}")
        print(f"Current Position: {curr_pos}")
        # Sleep pose
        self.SLEEP_X = 0.160
        self.SLEEP_Y = 0.00
        self.SLEEP_Z = 0.130
        self.SLEEP_R = 0.0

        self.last_command = None

    # Jog to safe home
    def go_to_sleep(self):
        print("[ROBOT] Going to sleep pose...")
        self.device.move_to(200.0,
                            0.0,
                            100.0,
                            0.0, 
                            wait=False)

    # Move to XYZ (meters)
    def point_at(self, x_m, y_m, z_m):
        print(f"[ROBOT] Pointing at {x_m:.3f}, {y_m:.3f}, {z_m:.3f}")
        self.last_command = (x_m, y_m, z_m)
        self.device.move_to(x_m * 1000, y_m * 1000, z_m * 1000, self.SLEEP_R)

    def emergency_stop(self):
        print("EMERGENCY STOP!")
        # pydobot does not support queue stop; no-op
        # Optionally turn off pump or motors
        # self.device.suck(False)
        pass

# ---------------------------------------------------------------------------
def load_calibration(calib_file=CALIB_FILE):
    with open(calib_file, "r") as f:
        data = yaml.safe_load(f)

    R = np.array(data["rotation_matrix"], dtype=float)
    t = np.array(data["translation_m"], dtype=float).reshape(3)

    print("Calibration loaded:")
    print("R=", R)
    print("t=", t)
    return R, t

# YOLO node creation remains unchanged

def create_yolo_node(pipeline, model_name=model_name, zoo_type=zoo_type, conf_thr=0.45, iou_thr=0.45):
    yolo = pipeline.create(dai.node.YoloDetectionNetwork)
    blob_path = blobconverter.from_zoo(name=model_name, shaves=3, zoo_type=zoo_type)
    yolo.setBlobPath(blob_path)
    yolo.setConfidenceThreshold(conf_thr)
    yolo.setIouThreshold(iou_thr)
    yolo.setNumClasses(80)
    yolo.setCoordinateSize(4)
    yolo.setAnchors([
        10, 13, 16, 30, 33, 23,
        30, 61, 62, 45, 59, 119,
        116, 90, 156, 198, 373, 326
    ])
    yolo.setAnchorMasks({"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
    yolo.setNumInferenceThreads(2)
    yolo.input.setBlocking(False)
    yolo.input.setQueueSize(1)
    return yolo

# Pipeline creation unchanged

def create_pipeline():
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(416, 416)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    mono_left = pipeline.createMonoCamera()
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    mono_right = pipeline.createMonoCamera()
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    yolo = create_yolo_node(pipeline)
    cam_rgb.preview.link(yolo.input)

    xout_rgb = pipeline.createXLinkOut(); xout_rgb.setStreamName("rgb"); cam_rgb.preview.link(xout_rgb.input)
    xout_det = pipeline.createXLinkOut(); xout_det.setStreamName("detections"); yolo.out.link(xout_det.input)
    xout_depth = pipeline.createXLinkOut(); xout_depth.setStreamName("depth"); stereo.depth.link(xout_depth.input)

    return pipeline

# Depth ROI measurement unchanged

def measure_distance(depth_frame_mm, cx, cy, roi_size=9, min_valid_mm=300, max_valid_mm=15000):
    h, w = depth_frame_mm.shape[:2]
    half = roi_size // 2
    x1 = max(0, cx - half); x2 = min(w - 1, cx + half)
    y1 = max(0, cy - half); y2 = min(h - 1, cy + half)
    roi = depth_frame_mm[y1:y2+1, x1:x2+1].astype(np.float32)
    roi = roi[(roi > 0) & (roi >= min_valid_mm) & (roi <= max_valid_mm)]
    return None if roi.size == 0 else float(np.median(roi)) / 1000.0 # convert mm -> m

# Frame transform

def camera_to_robot(P_cam, R, t):
        # Your YAML contains P_cam = R * P_robot + t
    # So invert it:
    return R.T @ (P_cam - t)

# Workspace clamp remains identical

def clamp_workspace(P_robot):
    x, y, z = P_robot
    r = np.sqrt(x**2 + y**2)
    if r < REACH_MIN and r > 1e-4:
        scale = REACH_MIN / r; x *= scale; y *= scale
    if r > REACH_MAX:
        scale = REACH_MAX / r; x *= scale; y *= scale
    z = max(Z_MIN, min(Z_MAX, z))
    return np.array([x, y, z], float)

# ---------------------------------------------------------------------------
# Main loop (unchanged except robot calls)
# ---------------------------------------------------------------------------
def main():
    R, t = load_calibration()
    robot = RobotInterface()
    robot.go_to_sleep()

    pipeline = create_pipeline()

    tracking_enabled = True
    current_target_index = 0
    current_target_class = TARGET_CLASSES[current_target_index]
    last_target_time = 0

    P_smooth = None

    frame_count = 0
    t0 = time.time()

    cv2.namedWindow("visual_servo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("visual_servo", 900, 600)

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue("rgb", 4, False)
        qDet = device.getOutputQueue("detections", 4, False)
        qDepth = device.getOutputQueue("depth", 4, False)

        print("[INFO] Visual servoing started.")

        while True:
            inRgb = qRgb.get(); inDet = qDet.get(); inDepth = qDepth.get()
            frame = inRgb.getCvFrame()
            depth_frame = inDepth.getFrame()

            h_rgb, w_rgb = frame.shape[:2]
            h_depth, w_depth = depth_frame.shape[:2]

            frame_count += 1
            fps = frame_count / (time.time() - t0)

            detections = inDet.detections
            best_det = None; best_conf = 0

            for det in detections:
                if det.label >= len(LABEL_MAP): continue
                label = LABEL_MAP[det.label]
                conf = det.confidence
                color = (0,255,0) if label == current_target_class else (100,100,100)

                x1 = int(det.xmin * w_rgb); y1 = int(det.ymin * h_rgb)
                x2 = int(det.xmax * w_rgb); y2 = int(det.ymax * h_rgb)
                x1 = max(0, min(x1, w_rgb-1)); x2 = max(0, min(x2, w_rgb-1))
                y1 = max(0, min(y1, h_rgb-1)); y2 = max(0, min(y2, h_rgb-1))

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cx_rgb = (x1 + x2)//2; cy_rgb = (y1 + y2)//2
                cv2.drawMarker(frame, (cx_rgb, cy_rgb), color, cv2.MARKER_CROSS, 8, 2)

                if label == current_target_class and conf > CONF_THRESH:
                    if conf > best_conf:
                        best_conf = conf
                        best_det = (cx_rgb, cy_rgb, conf)

            depth_str = "N/A"

            if best_det is not None:
                cx_rgb, cy_rgb, conf = best_det
                cx_depth = int(cx_rgb * w_depth / w_rgb)
                cy_depth = int(cy_rgb * h_depth / h_rgb)

                depth_m = measure_distance(depth_frame, cx_depth, cy_depth)
                if depth_m is not None:
                    depth_str = f"{depth_m:.2f} m"

                    X_cam = (cx_rgb - CX) * depth_m / FX
                    Y_cam = (cy_rgb - CY) * depth_m / FY
                    Z_cam = depth_m

                    P_cam = np.array([X_cam, Y_cam, Z_cam])
                    P_robot = camera_to_robot(P_cam, R, t)
                    P_robot = clamp_workspace(P_robot)

                    if P_smooth is None:
                        P_smooth = P_robot
                    else:
                        P_smooth = ALPHA * P_robot + (1-ALPHA) * P_smooth

                    last_target_time = time.time()

                    if tracking_enabled:
                        robot.point_at(P_smooth[0], P_smooth[1], P_smooth[2])

                    text = f"XYZ: ({P_smooth[0]:.3f}, {P_smooth[1]:.3f}, {P_smooth[2]:.3f})"
                    cv2.putText(frame, text, (10, h_rgb-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            now = time.time()
            if tracking_enabled and now - last_target_time > TARGET_LOST_TIMEOUT:
                print("Target lost → sleep")
                tracking_enabled = False
                robot.go_to_sleep()
                P_smooth = None

            cv2.putText(frame, f"FPS: {fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Tracking: {'ON' if tracking_enabled else 'OFF'}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0) if tracking_enabled else (0,0,255), 1)
            cv2.putText(frame, f"Target: {current_target_class}", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(frame, f"Depth: {depth_str}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            cv2.imshow("visual_servo", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                robot.emergency_stop()
                robot.go_to_sleep()
                break
            elif key == ord('t'):
                tracking_enabled = not tracking_enabled
                if not tracking_enabled:
                    robot.go_to_sleep()
            elif key in [ord('a'), ord('b'), ord('c')]:
                idx = TARGET_KEYS.index(chr(key))
                current_target_class = TARGET_CLASSES[idx]
                P_smooth = None

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
