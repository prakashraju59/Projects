"""
task2.py
Lane Boundary Detection, Tracking, and Lane Departure Warning (LDW)
"""

import os
import cv2
import glob
import numpy as np
from collections import deque
from task1 import load_calibration, undistort_image, get_perspective_transform

# ------------------ CONFIG ------------------
VIDEO_FOLDER = r"C:/Users/praka/python_project/active safety/mu_pilot/videos"
OUTPUT_FOLDER = r"C:/Users/praka/python_project/active safety/mu_pilot/lane_videos_task2"
CALIB_FILE = "camera_params.npz"

# Binary Thresholds (SAME AS TASK1)
S_THRESH = (90, 255)
L_THRESH = (150, 255)
SOBEL_THRESH = (25, 255)

# Detection Parameters
NWINDOWS = 9
MARGIN = 60
MINPIX = 50
LINE_THICKNESS = 12
DASHED_THRESHOLD_RATIO = 0.7
CLASSIFICATION_HISTORY_SIZE = 10
CLASSIFICATION_CONSENSUS = 0.6
XM_PER_PIX = 3.7 / 640
LDW_THRESHOLD_M = 0.20
# --------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def binary_from_warped(img_warp):
    """Binary thresholding for lane detection using task1 thresholds."""
    hls = cv2.cvtColor(img_warp, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)

    l_mask = cv2.inRange(hls[:, :, 1], L_THRESH[0], L_THRESH[1])
    s_mask = cv2.inRange(hls[:, :, 2], S_THRESH[0], S_THRESH[1])

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx)) if np.max(abs_sobelx) != 0 else np.zeros_like(abs_sobelx, dtype=np.uint8)
    sx_mask = cv2.inRange(scaled_sobel, SOBEL_THRESH[0], SOBEL_THRESH[1])

    white_lane = cv2.bitwise_and(l_mask, sx_mask)
    yellow_lane = cv2.bitwise_and(s_mask, sx_mask)
    combined = cv2.bitwise_or(white_lane, yellow_lane)

    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)


class LaneStatus:
    """Tracks classification history and polynomial fit for temporal smoothing."""
    def __init__(self):
        self.is_solid_history = deque([False] * CLASSIFICATION_HISTORY_SIZE, maxlen=CLASSIFICATION_HISTORY_SIZE)
        self.current_fit = None
        self.y_coords = None

    def update_fit(self, new_fit, new_y_coords):
        self.current_fit = new_fit
        self.y_coords = new_y_coords

    def get_classification(self, h):
        """Returns 'Solid' or 'Dashed' based on temporal history."""
        if self.y_coords is None or len(self.y_coords) < MINPIX:
            self.is_solid_history.append(False)
            return 'Dashed'

        num_bins = 20
        bin_height = h / num_bins
        active_bins = sum(1 for i in range(num_bins) 
                         if np.sum((self.y_coords >= i * bin_height) & 
                                  (self.y_coords < (i + 1) * bin_height)) > 5)

        is_solid = (active_bins / num_bins >= DASHED_THRESHOLD_RATIO)
        self.is_solid_history.append(is_solid)

        return 'Solid' if sum(self.is_solid_history) / CLASSIFICATION_HISTORY_SIZE >= CLASSIFICATION_CONSENSUS else 'Dashed'


def find_lane_pixels_sliding(binary_warp):
    """Sliding window lane detection."""
    h, w = binary_warp.shape[:2]
    histogram = np.sum(binary_warp[h//2:, :], axis=0)
    midpoint = w // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nonzero = binary_warp.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    window_height = h // NWINDOWS
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    for window in range(NWINDOWS):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        
        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= leftx_current - MARGIN) & (nonzerox < leftx_current + MARGIN)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= rightx_current - MARGIN) & (nonzerox < rightx_current + MARGIN)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left) > MINPIX:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > MINPIX:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 50 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 50 else None
    
    return left_fit, right_fit, lefty, righty


def calculate_lateral_distance(fit, w):
    """Calculate lateral distance from vehicle center to lane line (in meters)."""
    if fit is None:
        return None
    h = 480
    x_lane_pix = fit[0] * (h-1)**2 + fit[1] * (h-1) + fit[2]
    return (x_lane_pix - w / 2) * XM_PER_PIX


def draw_output_lines(binary_warp, left_fit, left_type, right_fit, right_type):
    """Draw lane lines on BEV: Green=Solid, Blue=Dashed."""
    h, w = binary_warp.shape[:2]
    out_img = cv2.cvtColor(binary_warp, cv2.COLOR_GRAY2BGR)
    ploty = np.linspace(0, h-1, h)

    for fit, line_type in [(left_fit, left_type), (right_fit, right_type)]:
        if fit is not None:
            fitx = fit[0] * ploty**2 + fit[1] * ploty + fit[2]
            pts = np.array([np.transpose(np.vstack([fitx, ploty]))], dtype=np.int32)
            color = (0, 255, 0) if line_type == 'Solid' else (255, 0, 0)
            cv2.polylines(out_img, pts, False, color, LINE_THICKNESS)

    return out_img


def draw_lane_overlay(original, left_fit, left_type, right_fit, right_type, Minv):
    """Overlay detected lanes on original frame."""
    h, w = original.shape[:2]
    warp_zero = np.zeros((h, w, 3), dtype=np.uint8)
    ploty = np.linspace(0, h - 1, h)

    left_x = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2] if left_fit is not None else None
    right_x = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2] if right_fit is not None else None

    # Fill lane area
    if left_x is not None and right_x is not None:
        pts_left = np.array([np.transpose(np.vstack([left_x if left_x[-1] < right_x[-1] else right_x, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x if left_x[-1] < right_x[-1] else left_x, ploty])))])
        cv2.fillPoly(warp_zero, np.int_([np.hstack((pts_left, pts_right))]), (0, 255, 0))

    # Draw lane lines
    for fit, line_type in [(left_fit, left_type), (right_fit, right_type)]:
        if fit is not None:
            fitx = fit[0] * ploty**2 + fit[1] * ploty + fit[2]
            pts = np.array([np.transpose(np.vstack([fitx, ploty]))], dtype=np.int32)
            color = (0, 200, 0) if line_type == 'Solid' else (255, 120, 30)
            cv2.polylines(warp_zero, pts, False, color, LINE_THICKNESS)

    newwarp = cv2.warpPerspective(warp_zero, Minv, (w, h))
    return cv2.addWeighted(original, 1.0, newwarp, 0.7, 0)


def process_video_file(video_path, out_folder, mtx, dist):
    """Process video file with lane detection and LDW."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = 640, 480
    M, Minv, _, _ = get_perspective_transform((h, w))

    out_path = os.path.join(out_folder, "LDW_" + os.path.basename(video_path))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))

    left_status, right_status = LaneStatus(), LaneStatus()
    print(f"[INFO] Processing: {os.path.basename(video_path)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (w, h))
        frame = undistort_image(frame, mtx, dist)
        binary = binary_from_warped(cv2.warpPerspective(frame, M, (w, h)))

        left_fit, right_fit, lefty, righty = find_lane_pixels_sliding(binary)
        if left_fit is not None:
            left_status.update_fit(left_fit, lefty)
        if right_fit is not None:
            right_status.update_fit(right_fit, righty)

        left_type = left_status.get_classification(h)
        right_type = right_status.get_classification(h)

        # LDW logic
        warning_text = ""
        left_offset = calculate_lateral_distance(left_status.current_fit, w)
        right_offset = calculate_lateral_distance(right_status.current_fit, w)

        if left_status.current_fit is not None and left_type == "Solid" and left_offset is not None:
            if abs(left_offset) < LDW_THRESHOLD_M:
                warning_text = "LDW: DRIFTING LEFT"
        
        if right_status.current_fit is not None and right_type == "Solid" and right_offset is not None:
            if right_offset < LDW_THRESHOLD_M:
                warning_text = "LDW: DRIFTING RIGHT"

        result = draw_lane_overlay(frame, left_status.current_fit, left_type, right_status.current_fit, right_type, Minv)
        mono_lines = draw_output_lines(binary, left_status.current_fit, left_type, right_status.current_fit, right_type)

        if warning_text:
            cv2.putText(result, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(mono_lines, warning_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        writer.write(result)
        cv2.imshow("1. Final Result (Original + Overlay + LDW)", result)
        cv2.imshow("2. BEV with Lines", mono_lines)
        cv2.imshow("3. Binary (Lane Extraction)", binary)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    mtx, dist = load_calibration(CALIB_FILE)
    videos = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    
    if not videos:
        print("No videos found in", VIDEO_FOLDER)
    else:
        for v in videos:
            process_video_file(v, OUTPUT_FOLDER, mtx, dist)
    
    print("\nAll video processing complete ✅")
