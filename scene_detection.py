import sieve
from pydantic import BaseModel
from typing import List

class Scene(BaseModel):
    start_seconds: float
    end_seconds: float
    start_timecode: str
    end_timecode: str
    scene_number: int
    start_frame: int
    end_frame: int

def scene_detection(
    video: sieve.File,
    threshold: float = 27.0,
    adaptive_threshold: bool = False,
) -> Scene:
    """
    :param video: The video to detect scenes in
    :param threshold: Threshold the average change in pixel intensity must exceed to trigger a cut. Only used if adaptive_threshold is False.
    :param adaptive_threshold: Whether to use adaptive thresholding, which compares the difference in content between adjacent frames without a fixed threshold, and instead a rolling average of adjacent frame changes. This can help mitigate false detections in situations such as fast camera motions.
    :return: A list of scenes
    """

    from scenedetect.detectors import ContentDetector, AdaptiveDetector
    from scenedetect.scene_manager import SceneManager
    from scenedetect.video_manager import VideoManager

    import cv2
    cap = cv2.VideoCapture(video.path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    video_manager = VideoManager([video.path])
    scene_manager = SceneManager()
    if adaptive_threshold:
        scene_manager.add_detector(AdaptiveDetector())
    else:
        scene_manager.add_detector(ContentDetector(threshold=threshold))

    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    # print("List of scenes obtained:")
    scenes = []
    for i, scene in enumerate(scene_list):
        print(
            "Scene %2d: Start %s / Frame %d, End %s / Frame %d"
            % (
                i + 1,
                scene[0].get_timecode(),
                scene[0].get_frames(),
                scene[1].get_timecode(),
                scene[1].get_frames(),
            )
        )

        start_time_in_seconds = scene[0].get_frames() / fps
        end_time_in_seconds = scene[1].get_frames() / fps
        yield Scene(
            start_seconds=start_time_in_seconds,
            end_seconds=end_time_in_seconds,
            scene_number=i + 1,
            start_timecode=scene[0].get_timecode(),
            end_timecode=scene[1].get_timecode(),
            start_frame=scene[0].get_frames(),
            end_frame=scene[1].get_frames()
        ).dict()

    video_manager.release()