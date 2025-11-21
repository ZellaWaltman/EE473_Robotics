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

# Highlight these tag IDs (but we still show all)
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
# Main
# -------------------------------------------------------
def main():
    at_detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False,
    )

    pipeline = create_pipeline()

    # Make the window resizable (optional, just for nicer viewing)
    cv2.namedWindow("apriltag_detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("apriltag_detector", 800, 800)

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

            h, w = frame.shape[:2]

            for det in detections:
                tag_id = det.tag_id
                corners = det.corners.astype(int)
                center = det.center.astype(int)
                cx, cy = int(center[0]), int(center[1])

                if tag_id in INTEREST_TAG_IDS:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cv2.polylines(frame, [corners], True, color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                R_cam_tag = det.pose_R
                t_cam_tag = det.pose_t.flatten()
                x, y, z = t_cam_tag

                roll, pitch, yaw = rot_to_euler_rpy(R_cam_tag)
                roll_deg, pitch_deg, yaw_deg = np.degrees([roll, pitch, yaw])
                margin = det.decision_margin

                pos_text  = f"ID {tag_id}  ({x*1000:.0f}, {y*1000:.0f}, {z*1000:.0f}) mm"
                conf_text = f"margin: {margin:.1f}"
                rpy_text  = f"rpy(deg): ({roll_deg:.0f}, {pitch_deg:.0f}, {yaw_deg:.0f})"

                # Base text position to the right of the tag
                base_x = cx + 5
                base_y = max(cy - 25, 10)

                # Clamp X so text stays inside the image
                # 180 is a rough max text width in pixels
                text_margin = 180
                if base_x > w - text_margin:
                    base_x = w - text_margin

                cv2.putText(frame, pos_text,  (base_x, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, conf_text, (base_x, base_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(frame, rpy_text,  (base_x, base_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("apriltag_detector", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
