#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
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

# Tag size in meters (38mmx38mm)
TAG_SIZE_M = 0.038

# Highlight these tag IDs (highlighted in green)
INTEREST_TAG_IDS = {0, 1, 2, 3}

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
    cam.setInterleaved(False) # 3 seperate RGB channels
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    # Send Stream "rgb" to camera
    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    cam.preview.link(xout.input) # Previews streamed out

    return pipeline

# -------------------------------------------------------
# Convert rot matrix -> roll/pitch/yaw (for report)
# -------------------------------------------------------
# Avoiding SciPy bc it is a large dependency & we want less compute

def rot_to_euler_rpy(R):
    # Check for gimbal lock
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2) # r00 = r10
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
    # Create Apriltag detector from:
    at_detector = Detector(
        families="tag36h11",
        nthreads=4, # 4 CPU threads for detections
        quad_decimate=2.0, # Downsample image by 2 for speed
        quad_sigma=0.0, # 0 Gaussian Blur
        refine_edges=True, # Refine Detected Edges (more accurate corners)
        decode_sharpening=0.25, # Sharpen Image
        debug=False, # Disable Debug Visualizations
    )

    pipeline = create_pipeline()

    # Make the preview window resizable (just for nicer viewing)
    cv2.namedWindow("apriltag_detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("apriltag_detector", 800, 800)

    # Load pipeline into OAK Camera
    with dai.Device(pipeline) as device:
        # Queue size = 4, will store 4 frames
        # maxSize=4, blocking=False avoids app stalling if one stream lags; old frames drop instead
        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)

        frame_count = 0 # Frame count for FPS computation
        t0 = time.time() # Start Time for FPS computation

        # Start Message
        print("[INFO] Starting AprilTag detector. Press 'q' to quit.")

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame() # 416x416 BGR image

            # Convert color -> grayscale for Apriltag detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apriltag Detector
            detections = at_detector.detect(
                gray,
                estimate_tag_pose=True, # Estimate rot & translation of each tag from cam intrinsics
                camera_params=[FX, FY, CX, CY], # Cam intrinsics
                tag_size=TAG_SIZE_M, # Apriltag size (38mmx38mm)
            )
            
            # FPS Computation
            frame_count += 1
            now = time.time()
            fps = frame_count / (now - t0)

            # Get image height & width from [height, width, channels] array
            h, w = frame.shape[:2]

            # For each detected Apriltag
            for det in detections:
                tag_id = det.tag_id # Get ID
                corners = det.corners.astype(int) # Corners for bounding box drawing
                center = det.center.astype(int) # Center point of tag
                cx, cy = int(center[0]), int(center[1]) # x & y coords of center point of tag

                # Highlight tags in green
                if tag_id in INTEREST_TAG_IDS:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                # Draw bounding box around tag & circle at center
                cv2.polylines(frame, [corners], True, color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # Tag Rot Matrices wrt Cam Coords
                R_cam_tag = det.pose_R # rot matrix of tag wrt cam frame
                t_cam_tag = det.pose_t.flatten() # translation vector of tag wrt cam frame
                x, y, z = t_cam_tag

                
                roll, pitch, yaw = rot_to_euler_rpy(R_cam_tag) # Convert to roll/pitch/yaw (rad)
                roll_deg, pitch_deg, yaw_deg = np.degrees([roll, pitch, yaw]) # Rad -> deg
                
                margin = det.decision_margin # Decision margin (confidence) from Apriltag Detection
                
                # Display Text
                pos_text  = f"ID {tag_id}  ({x*1000:.0f}, {y*1000:.0f}, {z*1000:.0f}) mm" # Tag ID & xyz Position (mm)
                conf_text = f"margin: {margin:.1f}" # Confidence margin
                rpy_text  = f"rpy(deg): ({roll_deg:.0f}, {pitch_deg:.0f}, {yaw_deg:.0f})" # Roll/pitch/yaw (deg)

                # Base text position to the right of the tag
                base_x = cx + 5
                base_y = max(cy - 25, 10)

                # Clamp x so text stays inside the image
                # ~180 = max text width in pixels
                text_margin = 180
                if base_x > w - text_margin:
                    base_x = w - text_margin

                # Draw Bounding Boxes & DISPLAY TEXT
                cv2.putText(frame, pos_text,  (base_x, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, conf_text, (base_x, base_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(frame, rpy_text,  (base_x, base_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show processed frame
            cv2.imshow("apriltag_detector", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
