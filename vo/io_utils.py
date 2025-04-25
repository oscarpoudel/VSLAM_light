import os
import numpy as np
import cv2

def load_calibration_matrix(calib_file, cam_id="P0"):
    """Load the intrinsic matrix (3x3) from KITTI-style calibration file."""
    if not os.path.isfile(calib_file):
        print(f"[WARN] Calibration file not found: {calib_file}")
        return np.eye(3)
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith(cam_id + ":"):
                vals = line.strip().split()[1:]
                P = np.array(list(map(float, vals))).reshape(3, 4)
                return P[:, :3]
    return np.eye(3)

def load_image_sequence(img_dir):
    """Return sorted list of image file paths from directory."""
    if not os.path.isdir(img_dir):
        print(f"[WARN] Image directory not found: {img_dir}")
        return []
    return sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])

def load_poses(pose_file):
    """Load list of 3x4 ground truth pose matrices from text file."""
    if not os.path.isfile(pose_file):
        print(f"[WARN] Pose file not found: {pose_file}")
        return []
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) != 12:
                continue
            pose = np.fromstring(line, sep=' ').reshape(3, 4)
            poses.append(pose)
    return poses
