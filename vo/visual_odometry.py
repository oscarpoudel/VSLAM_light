import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from .feature_utils import detect_features, track_features, filter_by_parallax
from .motion_utils import estimate_pose
from .io_utils import load_image_sequence, load_poses

class VisualOdometry:
    """
    Visual Odometry pipeline with Kalman filtering for smoother trajectory.
    """
    def __init__(self, img_dir, pose_file, K, dt=1.0):
        self.img_files = load_image_sequence(img_dir)
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.gt_poses = load_poses(pose_file)
        self.gt_positions = [pose[:, 3] for pose in self.gt_poses]

        self.cur_t = np.zeros((3, 1))
        self.cur_R = np.eye(3)
        self.traj = []

        self.prev_img = None
        self.prev_pts = None

        self.dt = dt
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.kf.F = F
        self.kf.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.kf.x = np.zeros(6)
        self.kf.P *= 1e-3
        self.kf.R = np.eye(3) * 1e-2
        self.kf.Q = np.eye(6) * 1e-4

    def process_frame(self, idx: int):
        img_file = self.img_files[idx]
        img = cv2.imread(img_file)
        if img is None:
            print(f"[WARN] Couldn't read {img_file}")
            return None, None

        if idx == 0:
            self.prev_img = img
            self.prev_pts = detect_features(img)
            raw_pos = self.cur_t.flatten()[:3]
            self.kf.x[:3] = raw_pos
            self.traj.append(raw_pos.reshape(3, 1))
            return img, None

        pts1, pts2 = track_features(self.prev_img, img, self.prev_pts)
        pts1, pts2 = filter_by_parallax(pts1, pts2, min_parallax=5.0)
        if len(pts1) < 8:
            print(f"[INFO] Frame {idx}: insufficient matches.")
            self.prev_img = img
            self.prev_pts = detect_features(img)
            self.traj.append(self.kf.x[:3].copy().reshape(3, 1))
            return img, None

        R, t = estimate_pose(pts1, pts2, self.K)
        scale = 1.0
        if idx < len(self.gt_positions):
            gt_current = self.gt_positions[idx]
            gt_prev = self.gt_positions[idx - 1]
            gt_dist = np.linalg.norm(gt_current - gt_prev)
            vo_dist = np.linalg.norm(t)
            if vo_dist > 1e-5:
                scale = gt_dist / vo_dist

        self.cur_t += scale * (self.cur_R @ t)
        self.cur_R = R @ self.cur_R
        raw_pos = self.cur_t.flatten()[:3]

        self.kf.predict()
        self.kf.update(raw_pos)
        filtered_pos = self.kf.x[:3].copy()

        self.traj.append(filtered_pos.reshape(3, 1))
        self.prev_img = img
        self.prev_pts = detect_features(img)

        return img, pts2

    def run_all(self):
        for i in range(len(self.img_files)):
            yield self.process_frame(i)