#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import blobconverter
import time
from collections import Counter
from ultralytics import YOLO

# Define Frame Size: 416x416
FRAME_SIZE = (416, 416)

# Model Config
# ----------------------------------------------------
DET_INPUT_SIZE = (416, 416)
model_name = "yolov5n_coco_416x416"
zoo_type = "depthai"

# COCO labels (80 classes)
# ----------------------------------------------------
label_map = [
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

# Target classes for assignment
TARGET_CLASSES = {"laptop", "cup", "apple", "bottle", "person"}


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
    yolo = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

    blob_path = blobconverter.from_zoo(
        name=model_name,
        shaves=3,
        zoo_type=zoo_type
    )

    yolo.setBlobPath(blob_path)

    # Spatial detection specific parameters
    yolo.setConfidenceThreshold(0.5)
    yolo.input.setBlocking(False)
    yolo.setBoundingBoxScaleFactor(0.5)
    yolo.setDepthLowerThreshold(100) # Min 10 centimeters
    yolo.setDepthUpperThreshold(5000) # Max 5 meters
    
    # Yolo specific parameters
    yolo.setConfidenceThreshold(conf_thr)
    yolo.setIouThreshold(iou_thr)
    yolo.setNumClasses(80)
    yolo.setCoordinateSize(4)

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
    yolo.input.setQueueSize(1)

    return yolo


# ----------------------------------------------------
# Create Pipeline
# ----------------------------------------------------
def create_pipeline():
    pipeline = dai.Pipeline()

    # RGB camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(FRAME_SIZE[0], FRAME_SIZE[1])
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Mono cameras
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
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # YOLO Node
    yolo = create_yolo_node(pipeline)
    cam_rgb.preview.link(yolo.input)
    stereo.depth.link(yolo.inputDepth)
    
    # Output Streams
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_det = pipeline.create(dai.node.XLinkOut)
    xout_det.setStreamName("detections")
    yolo.out.link(xout_det.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


# ----------------------------------------------------
# Measure depth (7×7 ROI median)
# ----------------------------------------------------
def measure_distance(depth_frame_mm, cx, cy, roi_size=7,
                     min_valid_mm=300, max_valid_mm=15000):

    h, w = depth_frame_mm.shape[:2]

    half = roi_size // 2
    x1 = max(0, cx - half)
    x2 = min(w - 1, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h - 1, cy + half)

    roi = depth_frame_mm[y1:y2+1, x1:x2+1].astype(np.float32)
    roi = roi[(roi > 0) & (roi >= min_valid_mm) & (roi <= max_valid_mm)]

    if roi.size == 0:
        return None

    return float(np.median(roi))


# ----------------------------------------------------
# Main
# ----------------------------------------------------
pipeline = create_pipeline()

with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue("detections", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

    frame_count = 0
    start_time = time.time()
    last_report_time = start_time
    class_counts = Counter()

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()
        inDepth = qDepth.get()

        frame = inRgb.getCvFrame()
        depthFrame = inDepth.getFrame()

        h_rgb, w_rgb = frame.shape[:2]
        h_depth, w_depth = depthFrame.shape[:2]

        frame_count += 1
        now = time.time()
        fps = frame_count / (now - start_time)

        detections = inDet.detections

        for det in detections:
            if det.label >= len(label_map):
                continue
            label = label_map[det.label]

            if label not in TARGET_CLASSES:
                continue

            x1 = int(det.xmin * w_rgb)
            y1 = int(det.ymin * h_rgb)
            x2 = int(det.xmax * w_rgb)
            y2 = int(det.ymax * h_rgb)

            x1 = max(0, min(x1, w_rgb - 1))
            x2 = max(0, min(x2, w_rgb - 1))
            y1 = max(0, min(y1, h_rgb - 1))
            y2 = max(0, min(y2, h_rgb - 1))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cx_rgb = int((x1 + x2) / 2)
            cy_rgb = int((y1 + y2) / 2)

            cx_depth = int(cx_rgb * w_depth / w_rgb)
            cy_depth = int(cy_rgb * h_depth / h_rgb)

            # each `det` is a SpatialDetection
            X = det.spatialCoordinates.x  # mm
            Y = det.spatialCoordinates.y  # mm
            Z = det.spatialCoordinates.z  # mm (distance forward)

            dist_m = Z / 1000.0 if Z != 0 else None
            if dist_m is None:
            	dist_str = "dist: N/A"
            else:
            	dist_str = f"dist: {dist_m:.2f} m"

            text = f"{label} {det.confidence*100:.1f}% {dist_str}"

            cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            class_counts[label] += 1

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("OAK-D YOLOv5n + Depth", frame)

        depth_vis = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow("Depth", depth_vis)

        if now - last_report_time > 5.0:
            avg_fps = frame_count / (now - start_time)
            print("----- Stats -----")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Detection counts: {dict(class_counts)}")
            last_report_time = now

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
