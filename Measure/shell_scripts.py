from homography_app.models import PetVideos
from homography_app.task import process_video_task

def schedule_all_videos():
    for video in PetVideos.objects.all():
        process_video_task(video.id)
