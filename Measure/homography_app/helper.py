import numpy as np
import cv2

DEFAULT_HSV = np.array([26, 116, 152], dtype=np.uint8)
TOL_H, TOL_S, TOL_V = 10, 50, 100

from scipy import interpolate, signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def distance_from_homography(pt1, pt2, H):
    # Convert to proper shape for cv2.perspectiveTransform → (N, 1, 2)
    pts = np.array([pt1, pt2], dtype=np.float32).reshape(-1, 1, 2)

    # Transform both points using the homography
    world_pts = cv2.perspectiveTransform(pts, H)

    # Compute Euclidean distance in world coordinates
    p1, p2 = world_pts[0, 0], world_pts[1, 0]
    distance = np.linalg.norm(p1 - p2)

    return float(distance)

def detect_biggest_jump(y, smooth_window=11, smooth_poly=2, start_thresh=-1.5, end_thresh=0):
    y_smooth = savgol_filter(y, window_length=smooth_window, polyorder=smooth_poly)
    dy = np.gradient(y_smooth)

    n_base = max(1, len(y_smooth) // 10)
    baseline = np.median(y_smooth[-n_base:])

    jump_start_idx = np.where(dy < start_thresh)[0]
    jump_end_idx = np.where((dy > end_thresh) & (y_smooth > baseline))[0]

    start, end = None, None
    max_jump_height = 0

    for s in jump_start_idx:
        e_candidates = jump_end_idx[jump_end_idx > s]
        if len(e_candidates) == 0:
            continue
        for e in e_candidates:
            jump_height = np.max(y_smooth[s:e + 1]) - np.min(y_smooth[s:e + 1])
            if jump_height > max_jump_height:
                max_jump_height = jump_height
                start, end = s, e

    return start, end


def detect_jump_parabola(frames, pixel_y, filename="jump_parabola.png"):
    y_smooth = savgol_filter(pixel_y, window_length=9, polyorder=2)

    apex_idx = np.argmin(y_smooth)

    start_idx, end_idx = apex_idx, apex_idx
    while start_idx > 0 and y_smooth[start_idx] <= y_smooth[start_idx - 1]:
        start_idx -= 1
    while end_idx < len(y_smooth) - 1 and y_smooth[end_idx] <= y_smooth[end_idx + 1]:
        end_idx += 1

    jump_frames = frames[start_idx:end_idx + 1]
    jump_y = pixel_y[start_idx:end_idx + 1]

    coeffs = np.polyfit(jump_frames, jump_y, 2)
    poly = np.poly1d(coeffs)

    fit_x = np.linspace(jump_frames[0], jump_frames[-1], 300)
    fit_y = poly(fit_x)

    plt.figure(figsize=(10, 6))
    plt.plot(frames, pixel_y, 'bo-', label='Original Y')
    plt.plot(frames, y_smooth, 'g--', label='Smoothed Y')
    plt.plot(jump_frames, jump_y, 'ro', label='Detected Jump Region')
    plt.plot(fit_x, fit_y, 'k-', label='Fitted Parabola')
    plt.xlabel('Frame')
    plt.ylabel('Pixel Y')
    plt.title('Jump Detection and Parabola Fit')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

    # Optional: Return info about the jump
    return {
        'start_frame': jump_frames[0],
        'end_frame': jump_frames[-1],
        'apex_frame': frames[apex_idx],
        'coefficients': coeffs
    }

def save_frame_vs_pixel_graph(pixel_y, filename="frame_vs_pixel_y.png"):
    plt.figure(figsize=(10, 6))
    frames = [i for i in range(len(pixel_y))]
    plt.plot(frames, pixel_y, label='Pixel Y Position', color='b')
    plt.xlabel('Frame Number')
    plt.ylabel('Pixel Y Position')
    plt.title('Frame vs Pixel Y Position')
    plt.grid(True)
    plt.legend()

    # Save the graph to a file
    plt.savefig(filename)
    plt.close()  # Close the plot to avoid display


# Function: world → image projection using cv2.perspectiveTransform
def world_to_image(points, H):
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)  # (N,1,2)
    projected = cv2.perspectiveTransform(pts, H)
    return projected.reshape(-1, 2).astype(int)

def filter_and_smooth(coords, window_size=5, threshold=5):
    coords = np.array(coords, dtype=np.float64)
    N = len(coords)

    # Step 1: Robust outlier detection using rolling median
    coords_filtered = coords.copy()
    for i in range(N):
        # Define local window boundaries
        start = max(0, i - window_size)
        end = min(N, i + window_size + 1)

        local_median = np.median(coords[start:end], axis=0)
        distance = np.linalg.norm(coords[i] - local_median)

        if distance > threshold:
            coords_filtered[i] = np.array([np.nan, np.nan])

    # Step 2: Interpolation over NaN points
    valid_mask = ~np.isnan(coords_filtered[:, 0])
    x_valid = np.where(valid_mask)[0]
    y_valid = coords_filtered[valid_mask]

    if len(x_valid) < 4:
        kind = 'linear'
    else:
        kind = 'cubic'

    interp_func_x = interpolate.interp1d(x_valid, y_valid[:, 0], kind=kind, fill_value="extrapolate")
    interp_func_y = interpolate.interp1d(x_valid, y_valid[:, 1], kind=kind, fill_value="extrapolate")

    x_all = np.arange(N)
    coords_interpolated = np.vstack((interp_func_x(x_all), interp_func_y(x_all))).T

    # Step 3: Smoothing with Savitzky-Golay filter
    smoothed_x = signal.savgol_filter(coords_interpolated[:, 0], window_length=7, polyorder=2, mode='nearest')
    smoothed_y = signal.savgol_filter(coords_interpolated[:, 1], window_length=7, polyorder=2, mode='nearest')

    smoothed_coords = np.vstack((smoothed_x, smoothed_y)).T

    return smoothed_coords

def shadow_removal(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def equalize_image(img, clahe):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel (enhances brightness, reduces shadows)
    l_eq = clahe.apply(l)

    # Merge back and convert to BGR
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced_frame = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return enhanced_frame

def merge_close_points(points, threshold=10):
    merged = []
    points = points.copy()
    while points:
        base = points.pop(0)
        close_pts = [base]
        remaining = []
        for pt in points:
            if np.linalg.norm(np.array(base) - np.array(pt)) < threshold:
                close_pts.append(pt)
            else:
                remaining.append(pt)
        avg_x = int(np.mean([p[0] for p in close_pts]))
        avg_y = int(np.mean([p[1] for p in close_pts]))
        merged.append((avg_x, avg_y))
        points = remaining
    return merged

import json

def save_homography_as_json(H, path='homography_app/homography_data/homography.json'):
    H_list = H.tolist()  # Convert NumPy array to a Python list
    data = {
        "homography_matrix": H_list
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Homography saved to {path}")


def load_homography_from_json(path='homography_app/homography_data/homography.json'):
    with open(path, 'r') as f:
        data = json.load(f)
        H = np.array(data["homography_matrix"], dtype=np.float32)
    return H


def detect_and_measure_image(image_path, output_path, homography):
    picked_hsv = np.array([26, 116, 152])
    tol_h, tol_s, tol_v = 10, 50, 50
    lower = np.array([max(picked_hsv[0] - tol_h, 0),
                      max(picked_hsv[1] - tol_s, 0),
                      max(picked_hsv[2] - tol_v, 0)])
    upper = np.array([min(picked_hsv[0] + tol_h, 179),
                      min(picked_hsv[1] + tol_s, 255),
                      min(picked_hsv[2] + tol_v, 255)])

    frame = cv2.imread(image_path)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_points = []

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detected_points.append((cx, cy))

    # Merge close points
    def merge_close_points(points, threshold=10):
        merged = []
        points = points.copy()
        while points:
            base = points.pop(0)
            close_pts = [base]
            remaining = []
            for pt in points:
                if np.linalg.norm(np.array(base) - np.array(pt)) < threshold:
                    close_pts.append(pt)
                else:
                    remaining.append(pt)
            avg_x = int(np.mean([p[0] for p in close_pts]))
            avg_y = int(np.mean([p[1] for p in close_pts]))
            merged.append((avg_x, avg_y))
            points = remaining
        return merged

    detected_points = merge_close_points(detected_points)

    display_img = frame.copy()

    if len(detected_points) == 2:
        p1 = np.array([[detected_points[0]]], dtype=np.float32)
        p2 = np.array([[detected_points[1]]], dtype=np.float32)

        wp1 = cv2.perspectiveTransform(p1, homography)[0][0]
        wp2 = cv2.perspectiveTransform(p2, homography)[0][0]

        dist_cm = np.linalg.norm(wp1 - wp2)

        # Draw
        cv2.circle(display_img, detected_points[0], 6, (0, 0, 255), -1)
        cv2.circle(display_img, detected_points[1], 6, (0, 0, 255), -1)
        cv2.line(display_img, detected_points[0], detected_points[1], (0, 255, 0), 2)

        mid_x = (detected_points[0][0] + detected_points[1][0]) // 2
        mid_y = (detected_points[0][1] + detected_points[1][1]) // 2
        cv2.putText(display_img, f"{dist_cm:.1f} cm", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imwrite(output_path, display_img)
        return dist_cm
    else:
        return None
