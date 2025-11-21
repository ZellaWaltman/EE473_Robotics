#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
from pupil_apriltags import Detector

# -------------------------------------------------------
# Camera intrinsics for 416x416 (approx for OAK-D Wide)
# -------------------------------------------------------
FX = 450.0
FY = 450.0
CX = 208.0
CY = 208.0

# Tag size in METERS
TAG_SIZE_M = 0.038

# Optional: only highlight these tag IDs (but we’ll still show all)
INTEREST_TAG_IDS = {0, 1, 2, 3}

# -------------------------------------------------------
# Build DepthAI pipeline: RGB preview at 416x416
# -------------------------------------------------------
def create_pipeline():
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setPreviewSize(416, 416)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # OAK-D color cam
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    return pipeline

# -------------------------------------------------------
# Convert rotation matrix to roll/pitch/yaw (optional)
# -------------------------------------------------------
def rot_to_euler_rpy(R):
    """
    R: 3x3 rotation matrix (camera->tag or tag->camera)
    Returns roll, pitch, yaw in radians.
    Convention: XYZ (roll around x, pitch around y, yaw around z)
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock case
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    # Initialize AprilTag detector
    at_detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=2.0,     # lower = more accurate, slower; higher = faster, less accurate
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False,
    )

    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)

        frame_count = 0
        t0 = time.time()

        print("[INFO] Starting AprilTag detector. Press 'q' to quit.")

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()   # 416x416 BGR image

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = at_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[FX, FY, CX, CY],
                tag_size=TAG_SIZE_M,
            )

            frame_count += 1
            now = time.time()
            fps = frame_count / (now - t0)

            # Draw detections
            for det in detections:
                tag_id = det.tag_id
                corners = det.corners.astype(int)
                center = det.center.astype(int)
                center_xy = (int(center[0]), int(center[1]))

                # Choose color: highlight your main tags
                if tag_id in INTEREST_TAG_IDS:
                    color = (0, 255, 0)   # green
                else:
                    color = (255, 0, 0)   # blue for other tags

                # Draw outline
                cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=2)

                # Draw center
                cv2.circle(frame, center_xy, 4, (0, 0, 255), -1)

                # Pose wrt camera
                R_cam_tag = det.pose_R          # 3x3
                t_cam_tag = det.pose_t.flatten()  # (3,) in meters

                x, y, z = t_cam_tag  # camera frame coordinates in meters

                # Roll/pitch/yaw in degrees (optional, for debugging)
                roll, pitch, yaw = rot_to_euler_rpy(R_cam_tag)
                roll_deg = np.degrees(roll)
                pitch_deg = np.degrees(pitch)
                yaw_deg = np.degrees(yaw)

                # Detection confidence: decision_margin
                margin = det.decision_margin

                # Text overlays (position in mm + margin)
                pos_text = f"ID {tag_id}  ({x*1000:.0f}, {y*1000:.0f}, {z*1000:.0f}) mm"
                conf_text = f"margin: {margin:.1f}"
                rpy_text = f"rpy(deg): ({roll_deg:.0f}, {pitch_deg:.0f}, {yaw_deg:.0f})"

                y0 = max(center_xy[1] - 25, 10)
                cv2.putText(frame, pos_text, (center_xy[0] + 5, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, conf_text, (center_xy[0] + 5, y0 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(frame, rpy_text, (center_xy[0] + 5, y0 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # Show FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("apriltag_detector - camera frame", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
