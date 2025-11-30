#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import yaml
from pupil_apriltags import Detector

# -------------------------------------------------------
# Camera intrinsics for 416x416 resolution
# -------------------------------------------------------
# focal lengths
FX = 450.0
FY = 450.0
# center coords
CX = 208.0
CY = 208.0

# Tag size in meters
TAG_SIZE_M = 0.038

# Tags used for calibration
TAG_IDS = [0, 1, 2, 3]

# Pose samples per tag
SAMPLES_PER_TAG = 30

# File paths
KNOWN_POSITIONS_YAML = "apriltag_known_positions.yaml"
OUTPUT_CALIB_YAML = "camera_robot_calibration.yaml"

# -------------------------------------------------------
# Load known tag positions in robot base frame
# -------------------------------------------------------
def load_tag_positions_robot():
    """
    Expects YAML like:

    tag_positions:
      0: [0.30, -0.20, 0.00]
      1: [0.30,  0.20, 0.00]
      2: [0.098, -0.20, 0.00]
      3: [0.098,  0.20, 0.00]

    tag_size_m: 0.038
    """
    with open(KNOWN_POSITIONS_YAML, "r") as f:
        data = yaml.safe_load(f)

    tag_positions = {int(k): np.array(v, dtype=float)
                     for k, v in data["tag_positions"].items()}

    # Override TAG_SIZE_M from file if present
    if "tag_size_m" in data:
        global TAG_SIZE_M
        TAG_SIZE_M = float(data["tag_size_m"])
        print(f"[INFO] Loaded tag_size_m = {TAG_SIZE_M} from YAML")

    return tag_positions

# -------------------------------------------------------
# Rigid transform solver: P_robot = R * P_cam + t
# -------------------------------------------------------
def solve_rigid_transform(P_cam, P_robot):
    """
    P_cam:   Nx3 points in camera frame
    P_robot: Nx3 corresponding points in robot base frame
    Returns: R (3x3), t (3,) Camera Frame -> Robot Frame
    """
    assert P_cam.shape == P_robot.shape
    N = P_cam.shape[0]
    assert N >= 3, "Need at least 3 non-collinear points for a stable transform"

    cam_mean = P_cam.mean(axis=0)
    robot_mean = P_robot.mean(axis=0)

    Pc = P_cam - cam_mean
    Pr = P_robot - robot_mean

    H = Pc.T @ Pr  # 3x3

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = robot_mean - R @ cam_mean

    # Camera Frame -> Robot Frame R & 
    return R, t

# Convert rot matrix -> roll/pitch/yaw (for report)
# - - - - - - - - - - - - - - - - - - - - - - - - 
# Avoiding SciPy bc it is a large dependency & we want less compute
def rot_to_euler_rpy(R):
    # Check for gimbal lock
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw

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
    cam.preview.link(xout.input)

    return pipeline

# -------------------------------------------------------
# Display Text
# -------------------------------------------------------
def put_text_outline(img, text, org,
                     font=cv2.FONT_HERSHEY_SIMPLEX,
                     font_scale=0.5,
                     color=(255, 255, 255),
                     thickness=1):
    # Black outline
    cv2.putText(img, text, org, font, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Main colored text
    cv2.putText(img, text, org, font, font_scale,
                color, thickness, cv2.LINE_AA)

# -------------------------------------------------------
# Main calibration routine
# -------------------------------------------------------
def main():
    tag_positions_robot = load_tag_positions_robot()

    # Make sure all TAG_IDS exist in YAML
    for tid in TAG_IDS:
        if tid not in tag_positions_robot:
            raise RuntimeError(f"Tag ID {tid} not found in {KNOWN_POSITIONS_YAML}")

    # Create Apriltag detector from:
    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False,
    )

    # DepthAI pipeline
    pipeline = create_pipeline()

    # Storage for camera-frame samples
    samples_cam = {tid: [] for tid in TAG_IDS}

    # Resizable window for visualization
    cv2.namedWindow("calibration_collection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calibration_collection", 800, 800)

    # Load pipeline into OAK Camera
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)

        # FPS tracking
        t0 = time.time()
        frame_count = 0

        print("[INFO] Collecting AprilTag samples for calibration...")
        print(f"       Required samples per tag: {SAMPLES_PER_TAG}")

        # --------------------------------
        # Main Loop
        # --------------------------------
        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame() # Convert RGB -> OpenCV BGR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[FX, FY, CX, CY],
                tag_size=TAG_SIZE_M,
            )

            frame_count += 1
            fps = frame_count / (time.time() - t0)

            # Draw detections and collect samples
            for det in detections:
                tid = int(det.tag_id)
                corners = det.corners.astype(int)
                center = det.center.astype(int)
                cx, cy = int(center[0]), int(center[1])

                color = (0, 255, 0) if tid in TAG_IDS else (255, 0, 0)
                cv2.polylines(frame, [corners], True, color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                if tid in TAG_IDS:
                    t_cam = det.pose_t.flatten()  # 3D position in camera frame (meters)
                    samples_cam[tid].append(t_cam)

            # Get height/width of RGB images
            h,w = frame.shape[:2]
            x = w - 150
            y = h - 10

            # Text overlay: sample counts (drawn at the bottom-left)
            line_height = 25
            block_height = len(TAG_IDS) * line_height

            y0 = h - block_height - 10  # start 10 px above bottom block

            for tid in TAG_IDS:
                n = len(samples_cam[tid])

                put_text_outline(frame, f"Tag {tid}: {n}/{SAMPLES_PER_TAG}", (10, y0),
                        font_scale=0.6, color=(255, 255, 0), thickness=2)
                y0 += line_height

                put_text_outline(frame, f"FPS: {fps:.1f}", (x, y),
                        font_scale=0.7, color=(0, 0, 255), thickness=2)

            cv2.imshow("calibration_collection", frame)

            # Check if there are enough samples for all tags
            if all(len(samples_cam[tid]) >= SAMPLES_PER_TAG for tid in TAG_IDS):
                print("[INFO] Collected enough samples for all tags.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[WARN] Calibration collection aborted by user.")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()
        
    # -------------------------------------------------------
    # Compute calibration from collected samples
    # -------------------------------------------------------
    
    print("[INFO] Computing calibration...")

    P_cam_list = []
    P_robot_list = []

    for tid in TAG_IDS:
        samples = np.array(samples_cam[tid])  # (N,3)
        mean_cam = samples.mean(axis=0)       # average position per tag in camera frame

        P_cam_list.append(mean_cam)
        P_robot_list.append(tag_positions_robot[tid])

        print(f"[DEBUG] Tag {tid}: camera mean = {mean_cam}, robot = {tag_positions_robot[tid]}")

    P_cam = np.vstack(P_cam_list)    # shape (N_tags, 3)
    P_robot = np.vstack(P_robot_list)

    R, t = solve_rigid_transform(P_cam, P_robot)
   
    P_robot_pred = (R @ P_cam.T).T + t  # transform each P_cam
    print("Robot Frame Prediciton: ", P_robot_pred)

    errors = np.linalg.norm(P_robot_pred - P_robot, axis=1) * 1000.0  # mm

    mean_err = float(errors.mean())
    max_err = float(errors.max())
    std_err = float(errors.std())

    print("[INFO] Calibration results:")
    print("R (rotation matrix, camera->robot) =")
    print(R)
    print("t (translation vector, camera->robot, meters) =")
    print(t)
    print(f"Errors per tag (mm): {errors}")
    print(f"Mean error: {mean_err:.2f} mm")
    print(f"Max error:  {max_err:.2f} mm")
    print(f"Std error:  {std_err:.2f} mm")

    # Euler angles
    roll, pitch, yaw = rot_to_euler_rpy(R)
    roll_deg, pitch_deg, yaw_deg = np.degrees([roll, pitch, yaw])
    
    # -------------------------------------------------------
    # Save calibration to YAML
    # -------------------------------------------------------
    calib_data = {
        "rotation_matrix": R.tolist(),
        "translation_m": t.tolist(),
        "euler_rpy_rad": [float(roll), float(pitch), float(yaw)],
        "euler_rpy_deg": [float(roll_deg), float(pitch_deg), float(yaw_deg)],
        "errors_mm": {
            "per_tag": {int(tid): float(err) for tid, err in zip(TAG_IDS, errors)},
            "mean": mean_err,
            "max": max_err,
            "std": std_err,
        },
        "meta": {
            "tag_ids_used": [int(tid) for tid in TAG_IDS],
            "samples_per_tag": {int(tid): len(samples_cam[tid]) for tid in TAG_IDS},
            "tag_size_m": TAG_SIZE_M,
            "camera_params": [FX, FY, CX, CY],
            "timestamp": time.time(),
            "description": "Camera-to-robot-base calibration via AprilTags",
        },
    }

    with open(OUTPUT_CALIB_YAML, "w") as f:
        yaml.dump(calib_data, f)

    print(f"[INFO] Calibration saved to {OUTPUT_CALIB_YAML}")

if __name__ == "__main__":
    main()
