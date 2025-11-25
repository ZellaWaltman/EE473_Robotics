#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import yaml
import math
from pupil_apriltags import Detector
from pydobot import Dobot
from serial.tools import list_ports
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

# AprilTag Info
# - - - - - - - - - - - - 
TAG_SIZE_M = 0.038 # Tag size in meters (38mmx38mm)
HAND_TAG_ID = 4 # Tag mounted on end-effector

# .yaml Files
# - - - - - - - - - - - - 
CAMERA_ROBOT_CALIB_YAML = "camera_robot_calibration.yaml"
HANDEYE_OUTPUT_YAML = "handeye_ee_tag_calibration.yaml"

# Data collection settings
# - - - - - - - - - - - - 
MIN_SAMPLES = 15 # minimum pose pairs
MAX_SAMPLES = 40 # safety cap
MIN_DECISION_MARGIN = 20 # AprilTag detection quality threshold

# -------------------------------------------------------
# Helper Functions: build / invert homogeneous transforms
# -------------------------------------------------------

# Build 4x4 homogeneous transform
# - - - - - - - - - - - - - - - - - - - - - - - - 
def make_T(R, t):
    # R: 3x3 rotation
    # t: 3-vector
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T # return 4x4 homogeneous transform

# Invert 4x4 rigid transform
# - - - - - - - - - - - - - - - - - - - - - - - - 
def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return make_T(R_inv, t_inv)
  
# Convert rot matrix -> roll/pitch/yaw (for report)
# - - - - - - - - - - - - - - - - - - - - - - - - 
# Avoiding SciPy bc it is a large dependency & we want less compute
def rot_to_euler_rpy(R):
    # Check for gimbal lock
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
      
    return roll, pitch, yaw

# -------------------------------------------------------
# Load Camera -> Robot calibration from Part 2
# -------------------------------------------------------
# camera_robot_calibration.yaml stored R, t such that:
#  P_robot = R * P_camera + t
# So R, t describes transform from camera frame -> robot base frame
# Compute both: T_base_camera (camera -> base) & T_camera_base  (base -> camera)

def load_camera_robot_calibration():
  
    with open(CAMERA_ROBOT_CALIB_YAML, "r") as f:
        data = yaml.safe_load(f)

    R_cr = np.array(data["rotation_matrix"], dtype=float)
    t_cr = np.array(data["translation_m"], dtype=float).reshape(3)

    # Camera -> Robot base
    T_base_camera = make_T(R_cr, t_cr)

    # Robot base -> Camera: inverse
    R_cb = R_cr.T
    t_cb = -R_cb @ t_cr
    T_camera_base = make_T(R_cb, t_cb)

    print("[CALIB] Loaded camera → robot-base transform (T_base_camera).")
    print("R (camera→base) =\n", R_cr)
    print("t (camera→base) =", t_cr)

    return T_camera_base, T_base_camera

# -------------------------------------------------------
# Robot Interface (Dobot Magician) - Read Pose
# -------------------------------------------------------
class RobotInterface:
    def __init__(self):
        # Auto-select Dobot serial port
        ports = [p.device for p in list_ports.comports()
                 if 'ttyUSB' in p.device or 'ttyACM' in p.device]
        if not ports:
            raise RuntimeError("No Dobot serial ports found.")
        port = ports[0]
        print(f"[ROBOT] Using port: {port}")

        # Connect to Dobot via pydobot
        self.device = Dobot(port=port, verbose=False)
        print("[ROBOT] Connected to Dobot via pydobot.")

    def get_T_base_ee(self):
        """
        Read current end-effector pose from Dobot in robot base frame,
        and convert to 4x4 homogeneous transform T_base_ee.

        pydobot.pose() typically returns (x, y, z, r, j1, j2, j3, j4)
        where x,y,z are in mm and r is end-effector rotation about Z in degrees.
        """
        pose = self.device.pose()
        x_mm, y_mm, z_mm, r_deg = pose[0], pose[1], pose[2], pose[3]

        # mm -> m
        t_be = np.array([x_mm, y_mm, z_mm], dtype=float) / 1000.0

        # Approximate rotation as pure yaw (about Z-axis)
        theta = math.radians(r_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        R_be = np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)

        T_base_ee = make_T(R_be, t_be)
        return T_base_ee

# -------------------------------------------------------
# Build DepthAI pipeline: RGB preview = 416x416
# -------------------------------------------------------
def create_pipeline():
    pipeline = dai.Pipeline()

    # RGB Camera
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(416, 416)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    # Send Stream "rgb" to camera
    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    cam.preview.link(xout.input) # Previews streamed out

    return pipeline

# -------------------------------------------------------
# AprilTag Detector setup
# -------------------------------------------------------
def create_apriltag_detector():
    # Create Apriltag detector from:
    detector = Detector(
        families="tag36h11",
        nthreads=4, # 4 CPU threads for detections
        quad_decimate=2.0, # Downsample image by 2 for speed
        quad_sigma=0.0, # 0 Gaussian Blur
        refine_edges=True, # Refine Detected Edges (more accurate corners)
        decode_sharpening=0.25, # Sharpen Image
        debug=False, # Disable Debug Visualizations
    )
    return detector

# -------------------------------------------------------
# Average rigid transform T_e_t from multiple samples
# -------------------------------------------------------
# Each sample i: T_e_t_i
# T_list: list of 4x4 transforms (end-effector -> tag)

def average_transform(T_list):
    if len(T_list) == 0:
        raise ValueError("No transforms to average")

    # Stack Rs and ts
    Rs = [T[:3, :3] for T in T_list]
    ts = [T[:3, 3] for T in T_list]

    # Average rotation using SVD on sum(R_i)
    M = np.zeros((3, 3), dtype=float)
    for R in Rs:
        M += R
    U, S, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        # Fix reflection
        Vt[2, :] *= -1
        R_avg = U @ Vt

    # Average translation
    t_avg = np.mean(ts, axis=0)
  
    # Return averaged T (rotation via SVD on sum of R_i, translation via mean)
    return make_T(R_avg, t_avg)

# -------------------------------------------------------
# Compute reprojection errors
# -------------------------------------------------------
# Each sample i has
#      T_c_e[i]  = T_camera_base * T_base_ee[i]
#      T_c_t[i]  = measured from AprilTag
#      T_ee_tag  = calibrated from averaging
# Predicted tag pose = T_c_t_pred[i] = T_c_e[i] * T_ee_tag
# compute position & orientation error

def compute_errors(T_camera_base, T_base_ee_list, T_camera_tag_list, T_ee_tag):
    pos_errors_mm = []
    ang_errors_deg = []

    for T_b_e, T_c_t_meas in zip(T_base_ee_list, T_camera_tag_list):
        # Camera -> EE from FK chain
        T_c_e = T_camera_base @ T_b_e

        # Predicted tag pose in camera frame
        T_c_t_pred = T_c_e @ T_ee_tag

        # Extract translations
        t_meas = T_c_t_meas[:3, 3]
        t_pred = T_c_t_pred[:3, 3]

        # Position error (mm)
        e_pos = np.linalg.norm(t_meas - t_pred) * 1000.0
        pos_errors_mm.append(e_pos)

        # Orientation error
        R_meas = T_c_t_meas[:3, :3]
        R_pred = T_c_t_pred[:3, :3]
        R_err = R_pred.T @ R_meas
        angle_rad = math.acos(max(-1.0, min(1.0, (np.trace(R_err) - 1.0) / 2.0)))
        angle_deg = math.degrees(angle_rad)
        ang_errors_deg.append(angle_deg)

    pos_errors_mm = np.array(pos_errors_mm)
    ang_errors_deg = np.array(ang_errors_deg)

    stats = {
        "pos_mean_mm": float(pos_errors_mm.mean()),
        "pos_max_mm":  float(pos_errors_mm.max()),
        "pos_std_mm":  float(pos_errors_mm.std()),
        "ang_mean_deg": float(ang_errors_deg.mean()),
        "ang_max_deg":  float(ang_errors_deg.max()),
        "ang_std_deg":  float(ang_errors_deg.std()),
        "pos_per_sample_mm": pos_errors_mm.tolist(),
        "ang_per_sample_deg": ang_errors_deg.tolist(),
    }

    return stats

# -------------------------------------------------------
# MAIN: Hand-eye calibration & realtime tracking
# -------------------------------------------------------
def log_SO3(R):
    """
    Log map from SO(3) -> so(3) (axis-angle as 3-vector).
    Returns a 3D vector omega such that exp(omega^) = R.
    """
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp for numerical safety
    theta = math.acos(cos_theta)

    if abs(theta) < 1e-8:
        # Very small rotation -> approx zero
        return np.zeros(3, dtype=float)

    # Skew-symmetric part
    wx = R[2, 1] - R[1, 2]
    wy = R[0, 2] - R[2, 0]
    wz = R[1, 0] - R[0, 1]
    axis = np.array([wx, wy, wz], dtype=float) / (2.0 * math.sin(theta))

    # Axis-angle vector: theta * axis
    return theta * axis


def solve_handeye_AX_XB(T_base_ee_list, T_camera_tag_list):
    """
    AX = X B solver (Park–Martin style) to estimate X = T_ee_tag.

    We follow the assignment's definitions:

        A_i = T_base_ee[i]^{-1} * T_base_ee[i+1]
        B_i = T_camera_tag[i] * T_camera_tag[i+1]^{-1]

    and solve A_i * X = X * B_i for X (4x4 homogeneous).
    """
    n = len(T_base_ee_list)
    if n < 2:
        raise ValueError("Need at least 2 poses to form relative motions (A_i, B_i).")

    # ------------------------------------------------------------------
    # Build relative motions A_i, B_i
    # ------------------------------------------------------------------
    A_list = []
    B_list = []
    for i in range(n - 1):
        T_b_e_i = T_base_ee_list[i]
        T_b_e_j = T_base_ee_list[i + 1]
        A = invert_T(T_b_e_i) @ T_b_e_j   # base frame EE motion
        A_list.append(A)

        T_c_t_i = T_camera_tag_list[i]
        T_c_t_j = T_camera_tag_list[i + 1]
        B = invert_T(T_c_t_i) @ T_c_t_j   # camera frame tag motion
        B_list.append(B)

    # ------------------------------------------------------------------
    # 1) Solve rotation part R_X using Park–Martin method
    # ------------------------------------------------------------------
    a_vecs = []
    b_vecs = []
    for A, B in zip(A_list, B_list):
        R_A = A[:3, :3]
        R_B = B[:3, :3]
        a_vecs.append(log_SO3(R_A))
        b_vecs.append(log_SO3(R_B))

    # We want R_X such that a_i ≈ R_X * b_i  (least-squares Wahba problem)
    M = np.zeros((3, 3), dtype=float)
    for a, b in zip(a_vecs, b_vecs):
        M += np.outer(a, b)

    U, S, Vt = np.linalg.svd(M)
    R_X = U @ Vt
    if np.linalg.det(R_X) < 0:
        # Fix possible reflection
        Vt[-1, :] *= -1.0
        R_X = U @ Vt

    # ------------------------------------------------------------------
    # 2) Solve translation part t_X from:
    #    A_i X = X B_i
    #
    #    => R_Ai t_X + t_Ai = R_X t_Bi + t_X
    #    => (R_Ai - I) t_X = R_X t_Bi - t_Ai
    # ------------------------------------------------------------------
    C_rows = []
    d_rows = []
    I = np.eye(3, dtype=float)

    for A, B in zip(A_list, B_list):
        R_A = A[:3, :3]
        t_A = A[:3, 3]
        R_B = B[:3, :3]
        t_B = B[:3, 3]

        C_rows.append(R_A - I)
        d_rows.append(R_X @ t_B - t_A)

    C = np.vstack(C_rows)           # shape (3 * (n-1), 3)
    d = np.vstack(d_rows).reshape(-1)  # shape (3 * (n-1),)

    # Least-squares solve for t_X
    t_X, *_ = np.linalg.lstsq(C, d, rcond=None)
    t_X = t_X.reshape(3)

    # Final X = T_ee_tag
    T_ee_tag = make_T(R_X, t_X)
    return T_ee_tag

# -------------------------------------------------------
# MAIN: Hand-eye calibration & realtime tracking
# -------------------------------------------------------
def main():
    # Load Part 2 calibration: camera <-> robot base
    T_camera_base, T_base_camera = load_camera_robot_calibration()

    # Init robot interface
    robot = RobotInterface()

    # Init AprilTag detector
    at_detector = create_apriltag_detector()

    # DepthAI pipeline
    pipeline = create_pipeline()

    # Storage for samples
    T_base_ee_list = []
    T_camera_tag_list = []
    T_ee_tag_samples = []

    # Load pipeline into OAK Camera
    with dai.Device(pipeline) as device:
        # Queue size = 4, will store 4 frames
        # maxSize=4, blocking=False avoids app stalling if one stream lags; old frames drop instead
        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)

        cv2.namedWindow("handeye_collection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("handeye_collection", 900, 600)

        print("[INFO] Hand-Eye Calibration - Data Collection")
        print(f"       Move the robot through diverse poses where Tag ID {HAND_TAG_ID} is visible.")
        print(f"       Press 'c' to capture a sample when the tag is detected.")
        print(f"       Need at least {MIN_SAMPLES} samples. Press 'q' to quit early.")

        # ----------------------------------------------------------------
        # Phase 1: Data Collection
        # ----------------------------------------------------------------
        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = at_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[FX, FY, CX, CY],
                tag_size=TAG_SIZE_M,
            )

            # Draw detections & highlight tag on EE
            tag_found = False
            best_det = None

            for det in detections:
                tag_id = det.tag_id
                corners = det.corners.astype(int)
                center = det.center.astype(int)
                cx, cy = int(center[0]), int(center[1])

                color = (0, 255, 0) if tag_id == HAND_TAG_ID else (255, 0, 0)
                cv2.polylines(frame, [corners], True, color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                margin = det.decision_margin
                txt = f"ID {tag_id} m={margin:.1f}"
                cv2.putText(frame, txt, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                if tag_id == HAND_TAG_ID and margin >= MIN_DECISION_MARGIN:
                    tag_found = True
                    best_det = det

            n_samples = len(T_base_ee_list)
            info_text = f"Samples: {n_samples}/{MIN_SAMPLES} (max {MAX_SAMPLES})"
            cv2.putText(frame, info_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            hint_text = "Press 'c' to capture, 'q' to quit."
            cv2.putText(frame, hint_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if not tag_found:
                cv2.putText(frame, "Tag not detected or low margin...",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)

            cv2.imshow("handeye_collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[WARN] User quit during data collection.")
                cv2.destroyAllWindows()
                return

            if key == ord('c') and tag_found and best_det is not None:
                # Read robot EE pose
                T_base_ee = robot.get_T_base_ee()

                # Tag pose in camera frame from AprilTag detection
                R_cam_tag = best_det.pose_R
                t_cam_tag = best_det.pose_t.flatten()  # meters
                T_camera_tag = make_T(R_cam_tag, t_cam_tag)

                T_base_ee_list.append(T_base_ee)
                T_camera_tag_list.append(T_camera_tag)

                print(f"[INFO] Captured sample {len(T_base_ee_list)}")

                if len(T_base_ee_list) >= MAX_SAMPLES:
                    print("[INFO] Reached MAX_SAMPLES, stopping collection.")
                    break

            # Stop automatically if enough samples
            if len(T_base_ee_list) >= MIN_SAMPLES:
                # You can choose to keep collecting more, but let's break here
                print("[INFO] Collected required number of samples.")
                break

        cv2.destroyAllWindows()

        if len(T_base_ee_list) < MIN_SAMPLES:
            print("[ERROR] Not enough samples collected for hand-eye calibration.")
            return

        # ----------------------------------------------------------------
        # Phase 2: Solve for T_ee_tag by averaging
        # ----------------------------------------------------------------
        print("[INFO] Computing hand-eye transform T_ee_tag via averaging...")

        T_ee_tag = solve_handeye_AX_XB(T_base_ee_list, T_camera_tag_list)

        R_ee_tag = T_ee_tag[:3, :3]
        t_ee_tag = T_ee_tag[:3, 3]

        print("[INFO] Estimated T_ee_tag (end-effector -> tag):")
        print("R_ee_tag =\n", R_ee_tag)
        print("t_ee_tag (m) =", t_ee_tag)

        # Euler angles of tag frame w.r.t EE frame
        roll, pitch, yaw = rot_to_euler_rpy(R_ee_tag)
        roll_deg, pitch_deg, yaw_deg = np.degrees([roll, pitch, yaw])
        print(f"Euler rpy (deg) tag wrt EE: roll={roll_deg:.1f}, pitch={pitch_deg:.1f}, yaw={yaw_deg:.1f}")

        # -------------------------------------------------------
        # Phase 3: Error analysis
        # -------------------------------------------------------
        print("[INFO] Computing reprojection errors...")
        stats = compute_errors(T_camera_base, T_base_ee_list, T_camera_tag_list, T_ee_tag)

        print("Position error (mm): mean={:.2f}, max={:.2f}, std={:.2f}".format(
            stats["pos_mean_mm"], stats["pos_max_mm"], stats["pos_std_mm"]
        ))
        print("Orientation error (deg): mean={:.2f}, max={:.2f}, std={:.2f}".format(
            stats["ang_mean_deg"], stats["ang_max_deg"], stats["ang_std_deg"]
        ))

        # -------------------------------------------------------
        # Save calibration to YAML
        # -------------------------------------------------------
        calib_data = {
            "T_ee_tag": {
                "rotation_matrix": R_ee_tag.tolist(),
                "translation_m": t_ee_tag.tolist(),
                "euler_rpy_deg": [float(roll_deg), float(pitch_deg), float(yaw_deg)],
            },
            "errors": stats,
            "meta": {
                "num_samples": len(T_base_ee_list),
                "hand_tag_id": HAND_TAG_ID,
                "tag_size_m": TAG_SIZE_M,
                "camera_params": [FX, FY, CX, CY],
                "camera_robot_calib_file": CAMERA_ROBOT_CALIB_YAML,
                "timestamp": time.time(),
                "description": "End-effector to AprilTag hand-eye calibration (Dobot + OAK-D).",
            },
        }

        with open(HANDEYE_OUTPUT_YAML, "w") as f:
            yaml.dump(calib_data, f)

        print(f"[INFO] Hand-eye calibration saved to {HANDEYE_OUTPUT_YAML}")

        # -------------------------------------------------------
        # Phase 4: Real-time End-Effector tracking
        # -------------------------------------------------------
        print("[INFO] Starting real-time end-effector tracking.")
        print("      'q' - quit")

        cv2.namedWindow("ee_tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ee_tracking", 900, 600)

        T_tag_ee = invert_T(T_ee_tag)  # tag -> ee

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = at_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[FX, FY, CX, CY],
                tag_size=TAG_SIZE_M,
            )

            T_c_e_vision = None
            T_c_e_fk = None

            # Get FK-based EE pose
            T_b_e = robot.get_T_base_ee()
            T_c_e_fk = T_camera_base @ T_b_e
            t_fk = T_c_e_fk[:3, 3]

            # Draw AprilTags and compute vision-based pose
            for det in detections:
                tag_id = det.tag_id
                corners = det.corners.astype(int)
                center = det.center.astype(int)
                cx, cy = int(center[0]), int(center[1])

                color = (0, 255, 0) if tag_id == HAND_TAG_ID else (255, 0, 0)
                cv2.polylines(frame, [corners], True, color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                margin = det.decision_margin
                txt = f"ID {tag_id} m={margin:.1f}"
                cv2.putText(frame, txt, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                if tag_id == HAND_TAG_ID and margin >= MIN_DECISION_MARGIN:
                    R_c_t = det.pose_R
                    t_c_t = det.pose_t.flatten()
                    T_c_t = make_T(R_c_t, t_c_t)

                    # T_camera_ee = T_camera_tag * T_tag_ee
                    T_c_e_vision = T_c_t @ T_tag_ee

            # Display numeric info
            if T_c_e_vision is not None:
                t_vis = T_c_e_vision[:3, 3]

                # Position error between vision & FK in camera frame
                pos_err_mm = np.linalg.norm(t_vis - t_fk) * 1000.0

                cv2.putText(frame,
                            f"EE Vision XYZ (m): ({t_vis[0]:.3f}, {t_vis[1]:.3f}, {t_vis[2]:.3f})",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
                cv2.putText(frame,
                            f"EE FK XYZ (m):     ({t_fk[0]:.3f}, {t_fk[1]:.3f}, {t_fk[2]:.3f})",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
                cv2.putText(frame,
                            f"Vision vs FK pos error: {pos_err_mm:.1f} mm",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
            else:
                cv2.putText(frame,
                            "Tag not detected for vision EE pose...",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)

            cv2.imshow("ee_tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Exiting EE tracking.")
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
