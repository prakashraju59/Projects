# filepath: c:\Users\praka\python_project\active safety\mu_pilot\image-process.py
# pyright: reportMissingImports=false
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ros_robot_controller_msgs.msg import BuzzerState
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from collections import deque

# =====================================================================
# CONFIGURATION (Copied from Task 1 & 2)
# =====================================================================
CALIB_FILE = "camera_params.npz"
# Image processing parameters
NWINDOWS = 9
MARGIN = 60
MINPIX = 50
# Lane classification parameters
DASHED_THRESHOLD_RATIO = 0.7
CLASSIFICATION_HISTORY_SIZE = 10
CLASSIFICATION_CONSENSUS = 0.6
# LDW parameters
XM_PER_PIX = 3.7 / 640  # Meters per pixel in x dimension
LDW_THRESHOLD_M = 0.20  # Lane departure warning threshold in meters
# Buzzer parameters
BUZZER_FREQ = 2800
BUZZER_DURATION = 0.1
BUZZER_REPEAT = 2
# =====================================================================


# =====================================================================
# HELPER FUNCTIONS (Ported from Task 1)
# =====================================================================
def load_calibration(calib_file=CALIB_FILE):
    """Load camera matrix and distortion coefficients from .npz file."""
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file {calib_file} not found. Make sure it's in the same folder.")
    data = np.load(calib_file)
    return data["mtx"], data["dist"]

def undistort_image(img, mtx, dist):
    """Undistort an image using the provided camera matrix and distortion coefficients."""
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return undistorted

def get_perspective_transform(frame_shape, src=None, dst=None):
    """Compute perspective transform matrix (M) and inverse (Minv)."""
    h, w = frame_shape
    if src is None:
        src = np.float32([[208, 210], [300, 210], [520, 285], [0, 285]])
    if dst is None:
        dst = np.float32([[100, 0], [w-100, 0], [w-100, h], [100, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def binary_from_warped(img_warp):
    """Create a binary image where lane pixels are white."""
    hls = cv2.cvtColor(img_warp, cv2.COLOR_BGR2HLS)
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    s_thresh = (90, 255)
    l_thresh = (120, 255)
    gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    max_val = np.max(abs_sobelx) if np.max(abs_sobelx) != 0 else 1.0
    scaled = np.uint8(255 * abs_sobelx / max_val)
    sobel_thresh = (30, 255)
    s_mask = np.zeros_like(s, dtype=np.uint8)
    s_mask[(s >= s_thresh[0]) & (s <= s_thresh[1])] = 1
    l_mask = np.zeros_like(l, dtype=np.uint8)
    l_mask[(l >= l_thresh[0]) & (l <= l_thresh[1])] = 1
    sx_mask = np.zeros_like(scaled, dtype=np.uint8)
    sx_mask[(scaled >= sobel_thresh[0]) & (scaled <= sobel_thresh[1])] = 1
    combined = np.zeros_like(s_mask, dtype=np.uint8)
    combined[((s_mask == 1) | (l_mask == 1)) & (sx_mask == 1)] = 255
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    _, binary = cv2.threshold(combined, 1, 255, cv2.THRESH_BINARY)
    return binary

# =====================================================================
# LANE TRACKING & LDW LOGIC (Ported from Task 2)
# =====================================================================
class LaneStatus:
    """Tracks temporal classification and polynomial fits for each lane."""
    def __init__(self):
        self.is_solid_history = deque([False]*CLASSIFICATION_HISTORY_SIZE, maxlen=CLASSIFICATION_HISTORY_SIZE)
        self.current_fit = None
        self.y_coords = None

    def update_fit(self, fit, y_coords):
        self.current_fit = fit
        self.y_coords = y_coords

    def get_classification(self, h):
        """Classify lane as Solid or Dashed based on vertical continuity."""
        if self.y_coords is None or len(self.y_coords) < MINPIX:
            self.is_solid_history.append(False)
            return 'Dashed'
        num_bins = 20
        bin_height = h / num_bins
        active_bins = sum(1 for i in range(num_bins) if np.sum(np.logical_and(self.y_coords >= i * bin_height, self.y_coords < (i + 1) * bin_height)) > 5)
        is_solid = (active_bins / num_bins >= DASHED_THRESHOLD_RATIO)
        self.is_solid_history.append(is_solid)
        solid_ratio = sum(self.is_solid_history) / CLASSIFICATION_HISTORY_SIZE
        return 'Solid' if solid_ratio >= CLASSIFICATION_CONSENSUS else 'Dashed'

def find_lane_pixels_sliding(binary_warp):
    """Sliding window lane pixel detection."""
    histogram = np.sum(binary_warp[binary_warp.shape[0]//2:, :], axis=0)
    h, w = binary_warp.shape[:2]
    midpoint = int(w // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nonzero = binary_warp.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    window_height = int(h // NWINDOWS)
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    for window in range(NWINDOWS):
        win_y_low, win_y_high = h - (window + 1) * window_height, h - window * window_height
        win_xleft_low, win_xleft_high = leftx_current - MARGIN, leftx_current + MARGIN
        win_xright_low, win_xright_high = rightx_current - MARGIN, rightx_current + MARGIN
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > MINPIX: leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > MINPIX: rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else []
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else []
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 50 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 50 else None
    return left_fit, right_fit, lefty, righty

def calculate_lateral_distance(fit, w, h):
    """Compute lateral offset in meters from image center."""
    if fit is None: return None
    x_lane = fit[0]*(h-1)**2 + fit[1]*(h-1) + fit[2]
    center_pix = w / 2
    offset_pix = x_lane - center_pix
    return offset_pix * XM_PER_PIX

def draw_lane_overlay(original, left_fit, right_fit, Minv):
    """Overlay detected lanes back onto original frame."""
    h, w = original.shape[:2]
    warp_zero = np.zeros((h, w, 3), dtype=np.uint8)
    ploty = np.linspace(0, h-1, h)
    if left_fit is not None and right_fit is not None:
        leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
        lane_pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(warp_zero, np.int_([lane_pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(warp_zero, Minv, (w, h))
    return cv2.addWeighted(original, 1.0, newwarp, 0.7, 0)

# =====================================================================
# ROS2 NODE IMPLEMENTATION
# =====================================================================
class LaneDepartureNode(Node):
    def __init__(self):
        super().__init__('lane_departure_node')
        self.bridge = CvBridge()
        self.get_logger().info("Loading camera calibration...")
        try:
            self.mtx, self.dist = load_calibration()
        except FileNotFoundError as e:
            self.get_logger().error(f"FATAL: {e}. Cannot proceed without calibration file.")
            rclpy.shutdown()
            return

        # Image dimensions (assuming standard 640x480)
        self.w, self.h = 640, 480
        self.M, self.Minv = get_perspective_transform((self.h, self.w))

        # Lane status trackers
        self.left_status = LaneStatus()
        self.right_status = LaneStatus()

        # ROS Publishers and Subscribers
        self.subscription = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
        self.buzzer_pub = self.create_publisher(BuzzerState, 'ros_robot_controller/set_buzzer', 1)
        # Add a publisher for the processed image
        self.processed_image_pub = self.create_publisher(Image, '~/processed_image', 1)
        self.get_logger().info("Lane Departure Warning node initialized and running.")

    def sound_buzzer(self):
        """Publishes a message to activate the buzzer."""
        msg = BuzzerState()
        # Corrected: The 'freq' field must be an integer.
        msg.freq = int(BUZZER_FREQ)
        msg.on_time = float(BUZZER_DURATION)
        msg.off_time = float(BUZZER_DURATION / 5)
        msg.repeat = BUZZER_REPEAT
        self.buzzer_pub.publish(msg)
        self.get_logger().warn('Buzzer activated!')

    def image_callback(self, ros_image):
        """Main processing pipeline for each incoming image."""
        try:
            # 1. Convert ROS Image to OpenCV frame
            frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            frame = cv2.resize(frame, (self.w, self.h))

            # 2. Undistort and Warp
            undistorted = undistort_image(frame, self.mtx, self.dist)
            warped = cv2.warpPerspective(undistorted, self.M, (self.w, self.h))

            # 3. Create Binary Image
            binary = binary_from_warped(warped)

            # 4. Find Lane Pixels and Fit Polynomial
            left_fit, right_fit, lefty, righty = find_lane_pixels_sliding(binary)
            if left_fit is not None: self.left_status.update_fit(left_fit, lefty)
            if right_fit is not None: self.right_status.update_fit(right_fit, righty)

            # 5. Classify Lanes (Solid/Dashed)
            left_type = self.left_status.get_classification(self.h)
            right_type = self.right_status.get_classification(self.h)

            # 6. LDW Logic (Corrected)
            ldw_warning = False
            warning_text = ""

            # Only proceed if both lanes have a valid fit
            if self.left_status.current_fit is not None and self.right_status.current_fit is not None:
                # Calculate the x-position of left and right lanes at the bottom of the image
                y_eval = self.h - 1
                left_x = self.left_status.current_fit[0]*y_eval**2 + self.left_status.current_fit[1]*y_eval + self.left_status.current_fit[2]
                right_x = self.right_status.current_fit[0]*y_eval**2 + self.right_status.current_fit[1]*y_eval + self.right_status.current_fit[2]

                # Calculate the center of the lane and the car's position
                lane_center_pix = (left_x + right_x) / 2
                car_center_pix = self.w / 2

                # Calculate the offset in meters
                offset_m = (car_center_pix - lane_center_pix) * XM_PER_PIX

                # Check for departure from a SOLID lane
                # Drifting right: car center is too close to the right lane
                if right_type == "Solid" and (right_x - car_center_pix) * XM_PER_PIX < LDW_THRESHOLD_M:
                    ldw_warning = True
                    warning_text = "LDW: DRIFTING RIGHT"
                # Drifting left: car center is too close to the left lane
                elif left_type == "Solid" and (car_center_pix - left_x) * XM_PER_PIX < LDW_THRESHOLD_M:
                    ldw_warning = True
                    warning_text = "LDW: DRIFTING LEFT"

            # 7. Sound Buzzer if needed
            if ldw_warning:
                self.sound_buzzer()

            # 8. Visualization
            result = draw_lane_overlay(undistorted, self.left_status.current_fit, self.right_status.current_fit, self.Minv)
            if ldw_warning:
                cv2.putText(result, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # --- REMOVE DISPLAY CODE THAT CAUSES CRASH ---
            # cv2.imshow("Lane Departure Warning", result)
            # cv2.imshow("Binary BEV", binary)
            # key = cv2.waitKey(1)
            # if key == ord('q') or key == 27:
            #     self.get_logger().info("Shutdown requested by user.")
            #     cv2.destroyAllWindows()
            #     rclpy.shutdown()

            # --- PUBLISH THE PROCESSED IMAGE INSTEAD ---
            try:
                processed_msg = self.bridge.cv2_to_imgmsg(result, "bgr8")
                self.processed_image_pub.publish(processed_msg)
            except Exception as e:
                self.get_logger().error(f"Could not publish processed image: {e}")


        except Exception as e:
            self.get_logger().error(f"Error in image processing: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LaneDepartureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()