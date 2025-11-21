#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import yaml
from pupil_apriltags import Detector

# -------------------------------------------------------
# Camera intrinsics for 416x416 (approx for OAK-D Wide)
# -------------------------------------------------------
FX = 450.0
FY = 450.0
CX = 208.0
CY = 208.0

# Tag size in meters (should match your other scripts)
TAG_SIZE_M = 0.038

# File paths
KNOWN_POSITIONS_YAML = "apriltag_known_positions.yaml"
CALIB_YAML = "camera_robot_calibration.yaml"

# -------------------------------------------------------
# Load known tag positions (robot frame)
# -------------------------------------------------------
def load_tag_positions_robot():
    with open(KNOWN_POSITIONS_YAML, "r") as f:
        data = yaml.safe_load(f)
    tag_positions = {
        int(k): np.array(v, dtype=float)
        for k, v in data["tag_positions"].items()
    }

    # Optional: sync tag size from file
    if "tag_size_m" in data:
        global TAG_SIZE_M
        TAG_SIZE_M = float(data["tag_size_m"])
        print(f"[INFO] Loaded tag_size_m = {TAG_SIZE_M} from YAML")

    return tag_positions

# -------------------------------------------------------
# Load calibration (R, t) camera → robot
# -------------------------------------------------------
def load_calibration():
    with open(CALIB_YAML, "r") as f:
        data = yaml.safe_load(f)

    R = np.array(data["rotation_matrix"], dtype=float)
    t = np.array(data["translation_m"], dtype=float)

    print("[INFO] Loaded calibration:")
    print("R =")
    print(R)
    print("t =", t)

    return R, t, data

# -------------------------------------------------------
# DepthAI pipeline: 416x416 RGB preview
# -------------------------------------------------------
def create_pipeline():
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setPreviewSize(416, 416)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    return pipeline

# -------------------------------------------------------
# Map error to color (BGR)
# -------------------------------------------------------
def error_to_color(err_mm: float):
    if err_mm < 10.0:
        return (0, 255, 0)      # green
    elif err_mm < 20.0:
        return (0, 255, 255)    # yellow
    else:
        return (0, 0, 255)      # red

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    tag_positions_robot = load_tag_positions_robot()
    R, t, calib_meta = load_calibration()

    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False,
    )

    pipeline = create_pipeline()

    cv2.namedWindow("calibration_auto_check", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calibration_auto_check", 800, 800)

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)

        frame_count = 0
        t0 = time.time()

        print("[INFO] Starting calibration auto-check. Press 'q' to quit.")

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[FX, FY, CX, CY],
                tag_size=TAG_SIZE_M,
            )

            frame_count += 1
            fps = frame_count / (time.time() - t0)

            per_frame_errors = []

            for det in detections:
                tid = int(det.tag_id)
                corners = det.corners.astype(int)
                center = det.center.astype(int)
                cx, cy = int(center[0]), int(center[1])

                # Pose in camera frame (meters)
                P_cam = det.pose_t.flatten()

                # Transform to robot frame using calibration
                P_robot_est = R @ P_cam + t

                # If we know this tag's true position, compute error
                if tid in tag_positions_robot:
                    P_robot_gt = tag_positions_robot[tid]
                    err_mm = float(np.linalg.norm(P_robot_est - P_robot_gt) * 1000.0)
                    per_frame_errors.append(err_mm)

                    color = error_to_color(err_mm)

                    # Draw bbox & center in color
                    cv2.polylines(frame, [corners], True, color, 2)
                    cv2.circle(frame, (cx, cy), 4, color, -1)

                    text = f"ID {tid} err: {err_mm:.1f} mm"
                    cv2.putText(frame, text, (cx + 5, max(cy - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    # Tag without GT: draw in white
                    color = (255, 255, 255)
                    cv2.polylines(frame, [corners], True, color, 2)
                    cv2.circle(frame, (cx, cy), 4, color, -1)
                    cv2.putText(frame, f"ID {tid} (no GT)",
                                (cx + 5, max(cy - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Aggregate error stats for this frame
            if per_frame_errors:
                mean_err = np.mean(per_frame_errors)
                max_err = np.max(per_frame_errors)
            else:
                mean_err = float('nan')
                max_err = float('nan')

            # Show stats at top-left
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"Mean err: {mean_err:.1f} mm", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Max err:  {max_err:.1f} mm", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("calibration_auto_check", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
