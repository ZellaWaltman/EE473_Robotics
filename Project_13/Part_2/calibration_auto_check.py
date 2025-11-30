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

# File paths
KNOWN_POSITIONS_YAML = "apriltag_known_positions.yaml"
CALIB_YAML = "camera_robot_calibration.yaml"

# -------------------------------------------------------
# Load known tag positions in robot base frame
# -------------------------------------------------------
def load_tag_positions_robot():
    with open(KNOWN_POSITIONS_YAML, "r") as f:
        data = yaml.safe_load(f)
    tag_positions = {
        int(k): np.array(v, dtype=float)
        for k, v in data["tag_positions"].items()
    }

    # Override TAG_SIZE_M from file if present
    if "tag_size_m" in data:
        global TAG_SIZE_M
        TAG_SIZE_M = float(data["tag_size_m"])
        print(f"[INFO] Loaded tag_size_m = {TAG_SIZE_M} from YAML")

    return tag_positions

# -------------------------------------------------------
# Load calibration (R, t) Camera Frame -> Robot Frame
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
# Main
# -------------------------------------------------------
def main():
    tag_positions_robot = load_tag_positions_robot()
    R, t, calib_meta = load_calibration()

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

    # Resizable window for visualization
    cv2.namedWindow("calibration_auto_check", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calibration_auto_check", 800, 800)

    # Load pipeline into OAK Camera
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)

        # FPS tracking
        t0 = time.time()
        frame_count = 0

        print("[INFO] Starting calibration auto-check. Press 'q' to quit.")

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

            per_frame_errors = []

            # Draw detections
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

            # - - - - - - - - - - - - - - - - - - - - - - - - -
            # Show stats at bottom-right w/ outlined text
            # - - - - - - - - - - - - - - - - - - - - - - - - -
            
            # Get height/width of RGB images
            h, w = frame.shape[:2]
            
            lines = [
                f"FPS: {fps:.1f}",
                f"Mean err: {mean_err:.1f} mm",
                f"Max err:  {max_err:.1f} mm",
            ]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            line_spacing = 4
            
            # Compute common line height
            _, text_size = cv2.getTextSize(lines[0], font, font_scale, thickness)
            line_height = text_size[1] + line_spacing
            
            y = h - 10  # start a bit above bottom edge
            
            # Draw from bottom to top so the block is tight to the bottom-right corner
            for text in reversed(lines):
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x = w - tw - 10  # 10 px from right edge
            
                color = (255, 255, 255)  # white text; outline makes it visible
                put_text_outline(frame, text, (x, y),
                                 font=font, font_scale=font_scale,
                                 color=color, thickness=thickness)
                y -= line_height

            cv2.imshow("calibration_auto_check", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
