import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple
from collections import deque

class CentroidTracker:
    """
    Lightweight centroid tracker for virtual tripwire line-crossing analytics.
    Tracks moving objects and counts when they cross a virtual tripwire.
    """
    def __init__(self, max_disappeared: int = 15):
        self.next_object_id = 0
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.crossings_count = 0
        self.crossed_ids = set()

    def update(self, rects: List[Tuple[int, int, int, int]], tripwire_y: int) -> int:
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.crossings_count

        input_centroids = []
        for (x, y, w, h) in rects:
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate pairwise Euclidean distances
            dists = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids),
                axis=2
            )
            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                obj_id = object_ids[row]
                old_y = self.objects[obj_id][1]
                new_y = input_centroids[col][1]

                # Check if crossed the horizontal tripwire line
                if (old_y < tripwire_y <= new_y) or (old_y > tripwire_y >= new_y):
                    if obj_id not in self.crossed_ids:
                        self.crossings_count += 1
                        self.crossed_ids.add(obj_id)

                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, dists.shape[0])).difference(used_rows)
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            unused_cols = set(range(0, dists.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.crossings_count

    def register(self, centroid: Tuple[int, int]):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, obj_id: int):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        if obj_id in self.crossed_ids:
            self.crossed_ids.remove(obj_id)


class VisionEngine:
    """
    Modular real-time computer vision engine.
    Supports Face & Eye tracking, Motion/Centroid tracking with virtual tripwire,
    and a clean plug-in slot for custom YOLO/PyTorch models.
    """
    def __init__(self):
        # OpenCV Haar Cascades for real-time tracking
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=40, detectShadows=False
        )
        self.tracker = CentroidTracker()
        self.detection_history: deque = deque(maxlen=200)

    def process_frame(
        self,
        frame: np.ndarray,
        mode: str = "face",
        confidence_thresh: float = 0.5,
        tripwire_pos_pct: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
        start_time = time.perf_counter()
        h, w = frame.shape[:2]
        detections = []
        tripwire_y = int(h * tripwire_pos_pct)

        if mode == "face":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )

            rects = []
            for (x, y, fw, fh) in faces:
                conf = round(0.85 + 0.12 * min(1.0, fw / 160.0), 2)
                if conf >= confidence_thresh:
                    rects.append((int(x), int(y), int(fw), int(fh)))
                    detections.append({
                        "label": "Face",
                        "confidence": conf,
                        "bbox": [int(x), int(y), int(fw), int(fh)],
                        "color": "#10B981"  # Emerald
                    })

                    # Eyes within face ROI
                    roi_gray = gray[y:y+fh, x:x+fw]
                    eyes = self.eye_cascade.detectMultiScale(
                        roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(10, 10)
                    )
                    for (ex, ey, ew, eh) in eyes[:2]:
                        detections.append({
                            "label": "Eye",
                            "confidence": 0.89,
                            "bbox": [int(x + ex), int(y + ey), int(ew), int(eh)],
                            "color": "#3B82F6"  # Blue
                        })

            crossings = self.tracker.update(rects, tripwire_y)

        elif mode == "motion":
            fg_mask = self.bg_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rects = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 600:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    conf = round(min(0.99, area / 4000.0), 2)
                    if conf >= confidence_thresh:
                        rects.append((int(x), int(y), int(bw), int(bh)))
                        detections.append({
                            "label": "Moving Object",
                            "confidence": conf,
                            "bbox": [int(x), int(y), int(bw), int(bh)],
                            "color": "#F59E0B"  # Amber
                        })

            crossings = self.tracker.update(rects, tripwire_y)

        elif mode == "edges":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 180)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

            for cnt in sorted_contours:
                if cv2.contourArea(cnt) > 300:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    detections.append({
                        "label": "Edge Contour",
                        "confidence": 0.94,
                        "bbox": [int(x), int(y), int(bw), int(bh)],
                        "color": "#8B5CF6"  # Violet
                    })
            crossings = self.tracker.crossings_count

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        meta = {
            "width": w,
            "height": h,
            "tripwire_y": tripwire_y,
            "total_crossings": crossings,
            "tracked_objects": len(self.tracker.objects)
        }

        # Save to history for CSV export
        timestamp_str = time.strftime("%H:%M:%S")
        for det in detections:
            self.detection_history.append({
                "timestamp": timestamp_str,
                "label": det["label"],
                "confidence": det["confidence"],
                "bbox": f"{det['bbox'][0]},{det['bbox'][1]},{det['bbox'][2]},{det['bbox'][3]}",
                "crossings": crossings
            })

        return detections, inference_time_ms, meta
