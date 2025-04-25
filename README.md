# Visual SLAM Project (vSLAM-lite)

This project implements a simple monocular visual odometry (vSLAM) pipeline using feature tracking and Kalman filtering for trajectory smoothing.

---

##  Project Structure

```
vslam_project/
├── main.py
├── requirements.txt
├── README.md
└── vo/
    ├── __init__.py
    ├── io_utils.py          # Load images, poses, calibration
    ├── feature_utils.py     # Feature detection, tracking, filtering
    ├── motion_utils.py      # Motion estimation, trajectory flipping
    └── visual_odometry.py   # VisualOdometry class with Kalman filtering
```

---

##  How to Run

1. Clone the repository:

```bash
git clone <your-repo-url>
cd vslam_project
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Prepare dataset:
- Set the dataset Directory in main.py
  - Images in: `./dataset/sequences/00/image_0`
  - Ground truth poses in: `./dataset/poses/00.txt`
  - Calibration file: `./dataset/sequences/00/calib.txt`

4. Run the main file:

```bash
python main.py
```

---
## outputs
![tracking points](https://github.com/oscarpoudel/VSLAM_light/blob/main/image/1.png)
![vslam](https://github.com/oscarpoudel/VSLAM_light/blob/main/image/2.png)


##  Live Outputs

- Window 1: Feature tracking overlay.
- Window 2: Real-time estimated trajectory.
- Window 3: Ground truth trajectory comparison.

---

##  Features

- Shi-Tomasi corner detection.
- Pyramidal Lucas-Kanade Optical Flow tracking.
- Essential matrix-based relative pose estimation.
- Constant-velocity model Kalman filtering.
- 2D live plotting of estimated vs ground-truth trajectory.
- Compatible with any KITTI-like dataset structure.

---

##  Dependencies

- OpenCV (cv2)
- NumPy
- Matplotlib
- FilterPy

(Handled automatically by `requirements.txt`.)

---

##  License

This project is open-sourced under the MIT License.

---

