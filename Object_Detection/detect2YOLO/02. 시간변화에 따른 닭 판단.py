import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time

# 모델 로드
model = YOLO("best_chick.pt")

# 비디오 파일 열기
cap = cv2.VideoCapture(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\detect2YOLO\datas\tile_r0_c2.mp4")

# 추적 중인 객체 정보 저장 (ID: [bbox, not_moving_frames])
tracked_objects = {}
object_id_counter = 0
movement_threshold = 5  # 움직임이 없다고 판단하는 픽셀 거리
not_moving_duration_threshold = 10  # 움직임 없음 지속 시간 (초)
frame_rate = 30  # 초기 프레임 레이트 추정 (실제 값은 동적으로 계산)
not_moving_frames_threshold = not_moving_duration_threshold * frame_rate

prev_frame_time = None

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

while cap.isOpened():
    current_time = time.time()
    if prev_frame_time is not None:
        frame_rate = 1 / (current_time - prev_frame_time)
        not_moving_frames_threshold = int(not_moving_duration_threshold * frame_rate)
    prev_frame_time = current_time

    ret, frame = cap.read()
    if not ret:
        break

    # 각 프레임에 대해 예측 수행
    results = model.predict(frame, imgsz=1280, conf=0.3, iou=0.5)
    current_detections = results[0].boxes.xyxy.cpu().numpy().astype(int)
    current_centers = []
    current_bboxes = []

    new_tracked_objects = {}
    current_detected_indices = list(range(len(current_detections)))
    matched_indices = set()

    # 기존 추적 객체와 현재 감지된 객체 매칭
    for obj_id, (prev_bbox, not_moving_frames) in list(tracked_objects.items()):
        best_match_index = -1
        max_iou = 0

        for i in current_detected_indices:
            current_bbox = current_detections[i]
            iou = calculate_iou(prev_bbox, current_bbox)
            if iou > max_iou:
                max_iou = iou
                best_match_index = i

        if best_match_index != -1:
            matched_indices.add(best_match_index)
            new_tracked_objects[obj_id] = [current_detections[best_match_index], not_moving_frames]
            current_detected_indices.remove(best_match_index)
        else:
            # 매칭 실패: 움직임 없음 카운터 유지
            new_tracked_objects[obj_id] = [prev_bbox, not_moving_frames + 1] # 프레임 단위로 카운트

    # 새로 감지된 객체 추가
    for index in current_detected_indices:
        new_tracked_objects[object_id_counter] = [current_detections[index], 0]
        object_id_counter += 1

    tracked_objects = new_tracked_objects

    # 결과 시각화 (프레임에 박스 그리기)
    annotated_frame = frame.copy()
    for obj_id, (bbox, not_moving_frames) in tracked_objects.items():
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)  # 기본은 초록색 (움직이는 상태)
        label = f"Chicken {obj_id}"

        # 움직임이 없는 시간 (프레임 수 기준) 확인
        if not_moving_frames >= not_moving_frames_threshold:
            color = (0, 0, 255)  # 10초 이상 움직임 없으면 파란색 (잠재적 폐사)
            label += " (Likely Dead)"
        elif not_moving_frames > 0:
            color = (0, 255, 255) # 움직임이 잠시 멈춘 경우 노란색
            label += f" (Not Moving: {not_moving_frames/frame_rate:.1f}s)"

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()