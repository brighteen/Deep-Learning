import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time

# 모델 로드
model = YOLO("best_chick.pt")

# 비디오 파일 열기
cap = cv2.VideoCapture(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\detect2YOLO\datas\tile_r0_c3.mp4")

# 추적 중인 객체 정보 저장 (ID: [bbox, stillness_duration])
tracked_objects = {}
object_id_counter = 0
movement_threshold = 15  # 움직임이 없다고 판단하는 픽셀 거리
stillness_increase_rate = 10 # 10초마다 증가하는 확률 (%)
time_interval_for_increase = 10 # 확률 증가 시간 간격 (초)

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
    time_elapsed = 0
    if prev_frame_time is not None:
        time_elapsed = current_time - prev_frame_time
    prev_frame_time = current_time

    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy() # 원본 프레임 저장
    dead_chicken_frame = np.zeros_like(frame) # 죽은 닭 위치만 표시할 프레임 생성

    # 각 프레임에 대해 예측 수행
    results = model.predict(frame, imgsz=1280, conf=0.3, iou=0.5)
    current_detections = results[0].boxes.xyxy.cpu().numpy().astype(int)

    new_tracked_objects = {}
    current_detected_indices = list(range(len(current_detections)))
    matched_indices = set()

    # 기존 추적 객체와 현재 감지된 객체 매칭
    for obj_id, (prev_bbox, stillness_duration) in list(tracked_objects.items()):
        best_match_index = -1
        max_iou = 0

        prev_center_x = (prev_bbox[0] + prev_bbox[2]) // 2
        prev_center_y = (prev_bbox[1] + prev_bbox[3]) // 2

        for i in current_detected_indices:
            current_bbox = current_detections[i]
            current_center_x = (current_bbox[0] + current_bbox[2]) // 2
            current_center_y = (current_bbox[1] + current_bbox[3]) // 2
            distance = np.sqrt((current_center_x - prev_center_x)**2 + (current_center_y - prev_center_y)**2)

            iou = calculate_iou(prev_bbox, current_bbox)

            # 움직임이 있다고 판단되면 stillness_duration 초기화
            if distance > movement_threshold:
                stillness_duration = 0
                if iou > max_iou:
                    max_iou = iou
                    best_match_index = i
            else:
                # 움직임이 없으면 경과 시간 추가
                stillness_duration += time_elapsed
                if iou > max_iou:
                    max_iou = iou
                    best_match_index = i

        if best_match_index != -1:
            matched_indices.add(best_match_index)
            new_tracked_objects[obj_id] = [current_detections[best_match_index], stillness_duration]
            current_detected_indices.remove(best_match_index)
        else:
            # 매칭 실패: 움직임 없음 시간 유지 (매칭 안됐으니 움직임 없다고 간주)
            new_tracked_objects[obj_id] = [prev_bbox, stillness_duration + time_elapsed]

    # 새로 감지된 객체 추가
    for index in current_detected_indices:
        new_tracked_objects[object_id_counter] = [current_detections[index], 0.0]
        object_id_counter += 1

    tracked_objects = new_tracked_objects

    # 결과 시각화 (원본 프레임)
    annotated_frame = original_frame.copy()
    for obj_id, (bbox, stillness_duration) in tracked_objects.items():
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)  # 기본은 초록색 (움직이는 상태)
        label = f"Chicken {obj_id}"

        death_probability = min(100, int((stillness_duration // time_interval_for_increase) * stillness_increase_rate))
        label += f" ({death_probability}%)"

        if death_probability >= 100:
            color = (0, 0, 255)  # 100% 확률이면 파란색 (죽은 닭)
        elif death_probability >= 20:
            color = (0, 0, 128) # 20% 이상이면 짙은 파란색
            cv2.rectangle(dead_chicken_frame, (x1, y1), (x2, y2), (0, 0, 128), 2)
            cv2.putText(dead_chicken_frame, f"Dead {obj_id} ({death_probability}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 128), 2)
        elif death_probability > 0:
            color = (0, 255, 255) # 움직임이 잠시 멈춘 경우 노란색

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 창 띄우기
    cv2.imshow("Original Detection", annotated_frame)
    cv2.imshow("Likely Dead Chickens", dead_chicken_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()