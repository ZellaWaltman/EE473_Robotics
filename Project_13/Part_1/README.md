# Part 1 – Real-Time Object Detection on OAK-D

This script implements **real-time object detection and depth measurement** using an **OAK-D camera** and a **YOLOv5n** model running directly on the Myriad X VPU via the DepthAI pipeline.

It is designed for the EE 473 / ECE 573 robotics assignment (Part 1).

---

## Features

- Uses **YOLOv5n (416×416)** precompiled blob from the **DepthAI model zoo** via `blobconverter`
- Runs inference fully **on the OAK-D** (low host CPU usage)
- Detects COCO objects and filters to target classes:
  - `laptop`, `cup`, `banana`, `bottle`, `person`
- Draws:
  - Bounding boxes
  - Class label + confidence
  - Crosshair at the depth measurement point
- Computes **depth** for each detection using:
  - Stereo depth map from the OAK-D
  - **7×7 ROI median filter** at the bounding box center
  - Displays distance as:
    - centimeters if `< 1m`
    - meters if `≥ 1m`
- Displays:
  - Real-time FPS overlay
  - Periodic console stats:
    - Average FPS
    - Detection counts per class

---

## Hardware & Software Requirements

### Hardware

- **OAK-D** (or OAK-D Wide) camera
- Host machine (e.g. **Raspberry Pi 4B** or desktop/laptop) with:
  - USB3 (preferred) or USB2 connection

### Software

- **Python 3.8+** (recommended)
- The Python packages listed in [`requirements.txt`](#installation)
- DepthAI drivers / udev rules installed (follow Luxonis docs if needed)
