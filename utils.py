from custom_types import VideoSegment, Frame, Box
from typing import List

def get_video_dimensions(path):
    import cv2
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return width, height

def get_video_length(path):
    import subprocess
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

def create_video_segments(file, scene_future, start_time=0, end_time=None, fps=30, original_video_length=None):
    scene_outputs = list(scene_future)

    video_segments = []

    '''
    CREATE VIDEO SEGMENTS
    '''
    for scene in scene_outputs:
        start_seconds = scene['start_seconds']
        end_seconds = scene['end_seconds']
        start_frame = scene['start_frame']
        end_frame = scene['end_frame']

        if end_time is not None and start_seconds > end_time:
            break

        if start_seconds < start_time and end_seconds < start_time:
            continue

        if start_seconds < start_time:
            start_seconds = start_time
            start_frame = int(start_seconds * fps)
        if end_time is not None and end_seconds > end_time:
            end_seconds = end_time
            end_frame = int(end_seconds * fps)

        video_segments.append(VideoSegment(path=file.path, start=start_seconds, end=end_seconds, start_frame=start_frame, end_frame=end_frame))

    if len(video_segments) == 0:
        if end_time is None:
            if original_video_length is None:
                original_video_length = get_video_length(file.path)
            end_time = original_video_length
        video_segments.append(VideoSegment(path=file.path, start=start_time, end=end_time))
    return video_segments

def track_boxes(
    frames: List[Frame],
    tracker = None,
    fps: float = 30,
):
    import supervision as sv
    if tracker is None:
        tracker = sv.ByteTrack(frame_rate=fps)
    new_frames = []
    for frame in frames:
        detections = frame.supervision_detection()
        new_detections = tracker.update_with_detections(detections)
        xyxys = new_detections.xyxy
        confidences = new_detections.confidence
        class_ids = new_detections.class_id
        tracker_ids = new_detections.tracker_id
        new_boxes = []
        for xyxy, confidence, class_id, tracker_id in zip(xyxys, confidences, class_ids, tracker_ids):
            # Ensure the box coordinates are within the frame
            x1 = max(0, xyxy[0])
            y1 = max(0, xyxy[1])
            x2 = min(frame.width, xyxy[2])
            y2 = min(frame.height, xyxy[3])
            
            new_boxes.append(Box(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                class_id=class_id,
                id=tracker_id,
            ))
        # convert back to boxes
        new_frames.append(Frame(
            boxes=new_boxes,
            number=frame.number,
            width=frame.width,
            height=frame.height,
        ))
    # remove all boxes that exist for less than 5 frames
    boxes_by_id = {}
    for frame in new_frames:
        for box in frame.boxes:
            if box.id not in boxes_by_id:
                boxes_by_id[box.id] = []
            boxes_by_id[box.id].append(box)
    output_frames = []
    for frame in new_frames:
        new_boxes = []
        for box in frame.boxes:
            if len(boxes_by_id[box.id]) >= min(15, len(new_frames)):
                new_boxes.append(box)
        output_frames.append(Frame(
            boxes=new_boxes,
            number=frame.number,
            width=frame.width,
            height=frame.height,
        ))
    return output_frames, tracker