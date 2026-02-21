"""
task1.py
Camera Calibration and Preprocessing Utilities
"""

import os
import glob
import cv2 as cv
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
CHESSBOARD_FOLDER = r"C:/Users/praka/python_project/active safety/mu_pilot/chessboard_images"
CHESSBOARD_BOARD_SIZE = (9, 13)
CALIB_FILE = "camera_params.npz"
OUTPUT_FOLDER_EXAMPLES = r"C:/Users/praka/python_project/active safety/mu_pilot/task1_examples"
UNDISTORTED_FOLDER = r"C:/Users/praka/python_project/active safety/mu_pilot/undistorted_chessboard"
CORNERS_FOLDER = r"C:/Users/praka/python_project/active safety/mu_pilot/corners_detected"

S_THRESH = (90, 255)
L_THRESH = (150, 255)
SOBEL_THRESH = (25, 255)
# ----------------------------

os.makedirs(OUTPUT_FOLDER_EXAMPLES, exist_ok=True)
os.makedirs(UNDISTORTED_FOLDER, exist_ok=True)
os.makedirs(CORNERS_FOLDER, exist_ok=True)


def calibrate_camera(image_folder, board_size=(9, 13), savefile=CALIB_FILE):
    """Find chessboard corners and calibrate camera."""
    w_corners, h_corners = board_size
    objp = np.zeros((w_corners * h_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w_corners, 0:h_corners].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    images = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
    if not images:
        raise FileNotFoundError(f"No chessboard images found in {image_folder}")

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in sorted(images):
        img = cv.imread(fname)
        if img is None:
            continue
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, board_size, None)
        
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

    if not objpoints:
        raise RuntimeError("No valid chessboard detections - cannot calibrate.")

    img_size = (gray.shape[1], gray.shape[0])
    rms, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    np.savez(savefile, mtx=mtx, dist=dist)
    
    print(f"[calibrate_camera] Saved to {savefile}, RMS error: {rms:.4f}")
    return mtx, dist, rms


def save_corners_detected(chessboard_folder, output_folder, board_size=(9, 13)):
    """Detect and draw chessboard corners on all images."""
    images = glob.glob(os.path.join(chessboard_folder, "*.jpg")) + glob.glob(os.path.join(chessboard_folder, "*.png"))
    if not images:
        print(f"[save_corners_detected] No images found")
        return
    
    print(f"[save_corners_detected] Processing {len(images)} images...")
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    detected_count = 0
    
    for fname in sorted(images):
        img = cv.imread(fname)
        if img is None:
            continue
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, board_size, None)
        
        img_output = img.copy()
        base_name = os.path.splitext(os.path.basename(fname))[0]
        
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(img_output, board_size, corners2, ret)
            cv.putText(img_output, f"Detected: {board_size[0]}x{board_size[1]}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            output_path = os.path.join(output_folder, f"{base_name}_corners.png")
            detected_count += 1
        else:
            cv.putText(img_output, "Detection FAILED", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output_path = os.path.join(output_folder, f"{base_name}_FAILED.png")
        
        cv.imwrite(output_path, img_output)
    
    print(f"[save_corners_detected] Detected {detected_count}/{len(images)} images → {output_folder}")


def load_calibration(calib_file=CALIB_FILE):
    """Load camera matrix and distortion coefficients."""
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")
    data = np.load(calib_file)
    return data["mtx"], data["dist"]


def undistort_image(img, mtx, dist):
    """Undistort an image using camera calibration."""
    h, w = img.shape[:2]
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv.undistort(img, mtx, dist, None, newcameramtx)


def save_undistorted_chessboards(chessboard_folder, output_folder, mtx, dist):
    """Undistort all chessboard images and save them."""
    images = glob.glob(os.path.join(chessboard_folder, "*.jpg")) + glob.glob(os.path.join(chessboard_folder, "*.png"))
    if not images:
        print(f"[save_undistorted_chessboards] No images found")
        return
    
    print(f"[save_undistorted_chessboards] Processing {len(images)} images...")
    
    for fname in sorted(images):
        img = cv.imread(fname)
        if img is None:
            continue
        
        undistorted = undistort_image(img, mtx, dist)
        base_name = os.path.splitext(os.path.basename(fname))[0]
        output_path = os.path.join(output_folder, f"{base_name}_undistorted.png")
        cv.imwrite(output_path, undistorted)
    
    print(f"[save_undistorted_chessboards] Saved {len(images)} images → {output_folder}")


def get_perspective_transform(frame_shape, src=None, dst=None):
    """Compute perspective transform matrix (M) and inverse (Minv)."""
    h, w = frame_shape if len(frame_shape) == 2 else frame_shape.shape[:2]

    src = np.float32(src) if src is not None else np.float32([[208, 210], [300, 210], [520, 285], [0, 285]])
    dst = np.float32(dst) if dst is not None else np.float32([[100, 0], [w-100, 0], [w-100, h], [100, h]])

    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)
    return M, Minv, src, dst


def warp_image(img, M, size=None):
    """Apply perspective warp using matrix M."""
    size = size or (img.shape[1], img.shape[0])
    return cv.warpPerspective(img, M, size, flags=cv.INTER_LINEAR)


def binary_from_warped(img_warp):
    """Binary thresholding for lane detection."""
    hls = cv.cvtColor(img_warp, cv.COLOR_BGR2HLS)
    gray = cv.cvtColor(img_warp, cv.COLOR_BGR2GRAY)

    l_mask = cv.inRange(hls[:, :, 1], L_THRESH[0], L_THRESH[1])
    s_mask = cv.inRange(hls[:, :, 2], S_THRESH[0], S_THRESH[1])

    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    max_sobel = np.max(abs_sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel) if max_sobel != 0 else np.zeros_like(abs_sobelx, dtype=np.uint8)
    sx_mask = cv.inRange(scaled_sobel, SOBEL_THRESH[0], SOBEL_THRESH[1])

    white_lane = cv.bitwise_and(l_mask, sx_mask)
    yellow_lane = cv.bitwise_and(s_mask, sx_mask)
    combined = cv.bitwise_or(white_lane, yellow_lane)

    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(combined, cv.MORPH_CLOSE, kernel, iterations=2)


def save_example_trapezoid_and_warp(example_frame_path, out_dir=OUTPUT_FOLDER_EXAMPLES):
    """Save example images: original with trapezoid, warped, and binary."""
    if not os.path.exists(example_frame_path):
        print("[save_example] Frame not found")
        return

    mtx, dist = load_calibration()
    frame = cv.imread(example_frame_path)
    und = undistort_image(frame, mtx, dist)
    h, w = und.shape[:2]
    M, Minv, src, dst = get_perspective_transform((h, w))

    vis = und.copy()
    cv.polylines(vis, [np.int32(src).reshape((-1, 1, 2))], True, (0, 0, 255), 3)
    cv.putText(vis, "SRC trapezoid (red)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    warped = warp_image(und, M, (w, h))
    binary_warped = binary_from_warped(warped)

    cv.imwrite(os.path.join(out_dir, "original_with_trapezoid.png"), vis)
    cv.imwrite(os.path.join(out_dir, "warped_example.png"), warped)
    cv.imwrite(os.path.join(out_dir, "binary_warped.png"), binary_warped)
    
    print(f"[save_example] Saved → {out_dir}")


if __name__ == "__main__":
    print("=== Task 1: Calibration & Preprocessing ===\n")
    try:
        save_corners_detected(CHESSBOARD_FOLDER, CORNERS_FOLDER, CHESSBOARD_BOARD_SIZE)
        
        print("\n[INFO] Calibrating camera...")
        mtx, dist, rms = calibrate_camera(CHESSBOARD_FOLDER, CHESSBOARD_BOARD_SIZE)
        
        print("\n[INFO] Saving undistorted images...")
        save_undistorted_chessboards(CHESSBOARD_FOLDER, UNDISTORTED_FOLDER, mtx, dist)
        
        print("\n[INFO] Generating examples from video...")
        video_files = glob.glob(r"C:/Users/praka/python_project/active safety/mu_pilot/videos/*.mp4")
        
        if video_files:
            cap = cv.VideoCapture(video_files[0])
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                sample_path = os.path.join(OUTPUT_FOLDER_EXAMPLES, "sample_frame.png")
                cv.imwrite(sample_path, frame)
                save_example_trapezoid_and_warp(sample_path)
        
        print("\n=== Task 1 Complete ===")
        print(f"✓ Corners: {CORNERS_FOLDER}")
        print(f"✓ Calibration: {CALIB_FILE}")
        print(f"✓ Undistorted: {UNDISTORTED_FOLDER}")
        print(f"✓ Examples: {OUTPUT_FOLDER_EXAMPLES}")
            
    except Exception as e:
        print(f"[ERROR] {e}")
