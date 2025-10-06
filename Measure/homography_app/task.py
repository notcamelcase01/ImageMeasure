import os
import logging
from background_task import background
from django.conf import settings
from django.core.files import File
import cv2
import json
import numpy as np
from .models import PetVideos, SingletonHomographicMatrixModel
from .helper import filter_and_smooth, DEFAULT_HSV, TOL_S, TOL_H, TOL_V, equalize_image, detect_biggest_jump, \
    distance_from_homography
import glob

logger = logging.getLogger('homography_app')


@background(schedule=0, remove_existing_tasks=True)
def process_video_task(petvideo_id):
    logger.info(f"[process_video_task] Starting processing for PetVideo ID: {petvideo_id}")

    try:
        video_obj = PetVideos.objects.get(id=petvideo_id)
    except PetVideos.DoesNotExist:
        logger.error(f"[process_video_task] PetVideo ID {petvideo_id} does not exist")
        return

    video_path = video_obj.file.path
    original_name = os.path.basename(video_obj.file.name)

    output_dir = os.path.join(settings.MEDIA_ROOT, 'post_processed_video')
    os.makedirs(output_dir, exist_ok=True)

    temp_output_path = os.path.join(output_dir, f"temp_{original_name}")
    final_output_path = os.path.join(output_dir, f"processed_{original_name}")

    try:
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        singleton = SingletonHomographicMatrixModel.load()

        if singleton.hsv_value:  # Will be {} if not set
            h = singleton.hsv_value.get('h', DEFAULT_HSV[0])
            s = singleton.hsv_value.get('s', DEFAULT_HSV[1])
            v = singleton.hsv_value.get('v', DEFAULT_HSV[2])
        else:
            h, s, v = DEFAULT_HSV

        lower = np.array([max(h - TOL_H, 0), max(s - TOL_S, 0), max(v - TOL_V, 0)])
        upper = np.array([min(h + TOL_H, 255), min(s + TOL_S, 255), min(v + TOL_V, 255)])

        trajectory = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                # --- Image Enhancement Step ---

            hsv_frame = cv2.cvtColor(equalize_image(frame, clahe), cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.medianBlur(mask, 5)

            # Saturation filter (optional but recommended)
            saturation_mask = hsv_frame[:, :, 1] > 64
            mask = mask & saturation_mask

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected_points = (0, 0)

            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if detected_points[1] < cy:
                        detected_points = (cx, cy)

            trajectory.append(detected_points)

        cap.release()

        cap = cv2.VideoCapture(video_path)
        traj_cnt = 0
        trajectory = filter_and_smooth(trajectory, threshold=10)

        start, end = detect_biggest_jump(trajectory[:, 1])
        pt1 = trajectory[start if start else 0, :]
        pt2 = trajectory[end if end else len(trajectory), :]
        trajectory = [tuple(map(int, point)) for point in trajectory]
        #sorted_points = sorted(trajectory, key=lambda p: p[1], reverse=True)


        folder_path = "/Users/notcamelcase/PycharmProjects/ImageMeasure/Measure/media/homograph"  # <-- change this
        files = glob.glob(os.path.join(folder_path, "homography_*.json"))

        if not files:
            raise FileNotFoundError("No file found matching homography_*.json")

        latest_file = max(files, key=os.path.getmtime)
        # Load JSON data
        with open(latest_file, "r") as f:
            H = np.array(json.load(f), dtype=np.float32)
        distance_ft = round(distance_from_homography(pt1, pt2, H), 2)
        img_line = np.array([[trajectory[start], trajectory[end]]], dtype=np.float32)
        world_line = cv2.perspectiveTransform(img_line, H)[0]
        p1, p2 = world_line
        vec = p2 - p1
        length = np.linalg.norm(vec)
        unit_vec = vec / length
        num_marks = int(length)  # one marker per foot
        scale_world = np.array([p1 + i * unit_vec for i in range(num_marks + 1)], dtype=np.float32).reshape(-1, 1, 2)
        H_inv = np.linalg.inv(H)
        scale_img = cv2.perspectiveTransform(scale_world, H_inv)
        scale_img = scale_img[:, 0, :].astype(int)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if start <= traj_cnt <= end:
                overlay = frame.copy()
                overlay[:] = (0, 0, 160)  # BGR red
                alpha = 0.3  # transparency factor
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.circle(frame, trajectory[traj_cnt], 20, (0, 255, 0), 2)
            traj_cnt += 1
            cv2.line(frame, trajectory[start], trajectory[end], (0, 0, 0), 2)

            for i, (x, y) in enumerate(scale_img):
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                    cv2.putText(frame, f"{i}ft", (x + 6, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            out.write(frame )
        cap.release()
        out.release()

        # --- ffmpeg encode ---
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', temp_output_path,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac', '-movflags', '+faststart', '-y',
            final_output_path
        ], check=True)

        # Save processed video to model
        with open(final_output_path, 'rb') as f:
            video_obj.processed_file.save(f"processed_{original_name}", File(f), save=False)

        video_obj.distance = distance_ft
        video_obj.is_video_processed = True
        video_obj.save()

        # Cleanup temporary files
        for path in [temp_output_path, final_output_path]:
            if os.path.exists(path):
                os.remove(path)

        logger.info(f"[process_video_task] Finished processing PetVideo ID: {petvideo_id}")

    except Exception as e:
        logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}")
