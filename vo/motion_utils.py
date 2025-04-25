import cv2
import numpy as np

def estimate_pose(pts1, pts2, K):
    """Estimate relative pose (R, t) using the Essential matrix and recoverPose."""
    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return np.eye(3), np.zeros((3,1))
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    if R is None or t is None:
        return np.eye(3), np.zeros((3,1))
    return R, t

R_flip_y = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

def flip_vo(vo_traj):
    """Apply 180-degree flip around Y-axis to match ground truth orientation."""
    return vo_traj @ R_flip_y.T
