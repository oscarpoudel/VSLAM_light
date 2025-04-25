import cv2
import matplotlib.pyplot as plt
import numpy as np
from vo.io_utils import load_calibration_matrix
from vo.visual_odometry import VisualOdometry
from vo.motion_utils import flip_vo

# if __name__ == "__main__":
img_dir = './dataset/sequences/00/image_0'
pose_file = './dataset/poses/00.txt'
calib_file = './dataset/sequences/00/calib.txt'

K = load_calibration_matrix(calib_file, cam_id='P0')
vo = VisualOdometry(img_dir, pose_file, K)

plt.ion()
fig_vo, ax_vo = plt.subplots()
vo_line, = ax_vo.plot([], [], 'b-', label='VO Trajectory')
ax_vo.set_title('VO Estimate')
ax_vo.grid()
ax_vo.axis('equal')
ax_vo.legend()

fig_gt, ax_gt = plt.subplots()
gt_line, = ax_gt.plot([], [], 'r-', label='Ground Truth')
ax_gt.set_title('Ground Truth')
ax_gt.grid()
ax_gt.axis('equal')
ax_gt.legend()

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", 640, 480)

gt_positions = np.array(vo.gt_positions)

for i, (img, pts2) in enumerate(vo.run_all()):
    if img is None:
        break

    vo_traj = np.array([p.flatten() for p in vo.traj])
    vo_traj = flip_vo(vo_traj)
    if vo_traj.shape[0] > 0:
        vo_line.set_xdata(vo_traj[:, 0])
        vo_line.set_ydata(vo_traj[:, 2])
        ax_vo.relim()
        ax_vo.autoscale_view(True, True, True)
        fig_vo.canvas.draw()
        fig_vo.canvas.flush_events()

    if i < gt_positions.shape[0]:
        gt_partial = gt_positions[:i+1]
        gt_line.set_xdata(gt_partial[:, 0])
        gt_line.set_ydata(gt_partial[:, 2])
        ax_gt.relim()
        ax_gt.autoscale_view(True, True, True)
        fig_gt.canvas.draw()
        fig_gt.canvas.flush_events()

    if pts2 is not None and pts2.shape[0] > 0:
        overlay = img.copy()
        for (x, y) in pts2:
            cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.imshow("Tracking", overlay)
    else:
        cv2.imshow("Tracking", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

plt.ioff()
plt.show()
cv2.destroyAllWindows()