import cv2
import numpy as np

def detect_features(img, max_corners=2000, quality_level=0.01, min_distance=7):
    """Detect good features (corners) to track using Shi-Tomasi algorithm."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    if corners is not None:
        return corners.reshape(-1, 2)
    return np.empty((0, 2))

def track_features(img1, img2, pts1):
    """Track features between two frames using pyramidal Lucas-Kanade optical flow."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    pts2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1.astype(np.float32), None)
    if pts2 is None or status is None:
        return np.empty((0, 2)), np.empty((0, 2))
    status = status.flatten()
    good1 = pts1[status == 1]
    good2 = pts2[status == 1]
    return good1, good2

def filter_by_parallax(pts1, pts2, min_parallax=5.0):
    """Keep only matches with displacement >= min_parallax."""
    if pts1.shape[0] == 0:
        return pts1, pts2
    dx = pts2[:, 0] - pts1[:, 0]
    dy = pts2[:, 1] - pts1[:, 1]
    dist = np.sqrt(dx**2 + dy**2)
    valid_mask = dist >= min_parallax
    pts1_f = pts1[valid_mask]
    pts2_f = pts2[valid_mask]
    return pts1_f, pts2_f
