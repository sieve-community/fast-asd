from pydantic import BaseModel, Field
from typing import List

class VideoSegment(BaseModel):
    path: str = Field(description="Path to the video segment")
    start: float = Field(description="Start time of the video segment")
    end: float = Field(description="End time of the video segment")
    start_frame: int = Field(default=None, description="Start frame of the video segment")
    end_frame: int = Field(default=None, description="End frame of the video segment")

    def dimensions(self):
        import cv2
        cap = cv2.VideoCapture(self.path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return width, height
  
    def duration(self):
        return self.end - self.start
    
    def fps(self):
        import cv2
        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps

class Box(BaseModel):
    class_id: int = Field(description="Class ID of the subject")
    confidence: float = Field(description="Confidence of the subject")
    x1: int = Field(description="X1 coordinate of the bounding box")
    y1: int = Field(description="Y1 coordinate of the bounding box")
    x2: int = Field(description="X2 coordinate of the bounding box")
    y2: int = Field(description="Y2 coordinate of the bounding box")
    id: int = Field(default=None, description="ID of the subject")
    metadata: dict = Field(default=None, description="Metadata of the subject")

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
    
    def width(self):
        return self.x2 - self.x1
    
    def height(self):
        return self.y2 - self.y1
    
    def overlap(self, box):
        return not (self.x2 < box.x1 or self.x1 > box.x2 or self.y2 < box.y1 or self.y1 > box.y2)
    
    def distance(self, box):
        from math import sqrt
        return sqrt((self.center()[0] - box.center()[0]) ** 2 + (self.center()[1] - box.center()[1]) ** 2)
    
    def overlap_percentage(self, box):
        if not self.overlap(box):
            return 0
        intersection_area = max(0, min(self.x2, box.x2) - max(self.x1, box.x1)) * max(0, min(self.y2, box.y2) - max(self.y1, box.y1))
        return intersection_area / max(self.area(), 1)
    
    def contains_point(self, point):
        x, y = point
        return x >= self.x1 and x <= self.x2 and y >= self.y1 and y <= self.y2
    
    def iou(self, box):
        if not self.overlap(box):
            return 0
        intersection_area = max(0, min(self.x2, box.x2) - max(self.x1, box.x1)) * max(0, min(self.y2, box.y2) - max(self.y1, box.y1))
        union_area = self.area() + box.area() - intersection_area
        return intersection_area / union_area

class Frame(BaseModel):
    number: int = Field(description="Frame number")
    width: int = Field(description="Width of the frame")
    height: int = Field(description="Height of the frame")
    boxes: List[Box] = Field(description="List of bounding boxes in the frame")

    def boxes_by_class(self, class_name):
        return [box for box in self.boxes if box.name == class_name]

    def area(self):
        return self.width * self.height
            
    def supervision_detection(self):
        import supervision as sv
        import numpy as np
        box_list = [[box.x1, box.y1, box.x2, box.y2] for box in self.boxes]
        confidence_list = [box.confidence for box in self.boxes]
        class_id_list = [box.class_id for box in self.boxes]
        if len(box_list) == 0:
            box_list = np.empty((0, 4))
            confidence_list = np.empty(0)
            class_id_list = np.empty(0)
        return sv.Detections(
            xyxy=np.array(box_list),
            confidence=np.array(confidence_list),
            class_id=np.array(class_id_list),
        )
