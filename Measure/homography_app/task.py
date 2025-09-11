import os
import logging
from background_task import background
from django.conf import settings
from django.core.files import File
import cv2
import numpy as np
from .models import PetVideos
from .helper import merge_close_points, DEFAULT_HSV, TOL_S, TOL_H, TOL_V

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

    # Ensure output folder exists
    output_dir = os.path.join(settings.MEDIA_ROOT, 'post_processed_video')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, original_name)

    try:
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = DEFAULT_HSV
            lower = np.array([max(h - TOL_H, 0), max(s - TOL_S, 0), max(v - TOL_V, 0)])
            upper = np.array([min(h + TOL_H, 179), min(s + TOL_S, 255), min(v + TOL_V, 255)])
            mask = cv2.inRange(hsv_frame, lower, upper)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected_points = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_points.append((cx, cy))

            detected_points = merge_close_points(detected_points, threshold=40)
            for pt in detected_points[:4 if len(detected_points) > 4 else len(detected_points)]:
                cv2.circle(frame, pt, 15, (0, 0, 255), -1)

            out.write(frame)

        cap.release()
        out.release()

        # Example distance calculation
        video_obj.distance = 10.9

        # Save processed video to model
        with open(output_path, 'rb') as f:
            video_obj.processed_file.save(original_name, File(f), save=True)

        # Delete temporary processed file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"[process_video_task] Temporary file deleted: {output_path}")
            except Exception as e:
                logger.warning(f"[process_video_task] Could not delete temp file: {e}")

        video_obj.save()
        logger.info(f"[process_video_task] Finished processing PetVideo ID: {petvideo_id}")

    except Exception as e:
        logger.error(f"[process_video_task] Error processing PetVideo ID {petvideo_id}: {e}")
