import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import defaultdict

# 모델 로드
model = YOLO("best_chick.pt")

# 비디오 파일 열기
cap = cv2.VideoCapture(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\detect2YOLO\datas\tile_r0_c1.mp4")

# 추적 중인 객체 정보 저장 (ID: [bbox, center_history])
tracked_objects = {}
object_id_counter = 0
center_history_length = 60 # 중심점 이동 기록 길이 (프레임 수)
low_frequency_threshold = 5 # 낮은 주파수 성분으로 간주할 최대 주파수 인덱스
motion_energy_threshold = 50 # 움직임 에너지 임계값 (조정 필요)

def calculate_iou(box1, box2):
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
    box2_area = (x2_2 - x1_2) * (y2_2 - y2_2)
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=1280, conf=0.3, iou=0.5)
    current_detections = results[0].boxes.xyxy.cpu().numpy().astype(int)

    new_tracked_objects = {}
    current_detected_indices = list(range(len(current_detections)))
    matched_indices = set()

    for obj_id, (prev_bbox, center_history) in list(tracked_objects.items()):
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

            if iou > max_iou:
                max_iou = iou
                best_match_index = i

        if best_match_index != -1:
            current_bbox = current_detections[best_match_index]
            current_center = ((current_bbox[0] + current_bbox[2]) // 2, (current_bbox[1] + current_bbox[3]) // 2)
            new_center_history = center_history[-center_history_length + 1:] + [current_center]
            new_tracked_objects[obj_id] = [current_bbox, new_center_history]
            matched_indices.add(best_match_index)
            if best_match_index in current_detected_indices:
                current_detected_indices.remove(best_match_index)
        else:
            # 매칭 실패: 이전 중심점 기록 유지
            new_tracked_objects[obj_id] = [prev_bbox, center_history]

    for index in current_detected_indices:
        bbox = current_detections[index]
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        new_tracked_objects[object_id_counter] = [bbox, [center]]
        object_id_counter += 1

    tracked_objects = new_tracked_objects

    annotated_frame = frame.copy()
    dead_chicken_frame = np.zeros_like(frame)

    for obj_id, (bbox, center_history) in tracked_objects.items():
        x1, y1, x2, y2 = bbox
        is_dead = False

        if len(center_history) >= center_history_length:
            center_x_history = [c[0] for c in center_history]
            center_y_history = [c[1] for c in center_history]

            fft_x = np.fft.fft(center_x_history)
            fft_y = np.fft.fft(center_y_history)

            # 낮은 주파수 성분 이후의 에너지 합산 (DC 성분 제외)
            energy_x = np.sum(np.abs(fft_x[low_frequency_threshold:]))
            energy_y = np.sum(np.abs(fft_y[low_frequency_threshold:]))
            total_energy = energy_x + energy_y

            if total_energy < motion_energy_threshold:
                is_dead = True

        if is_dead:
            color = (0, 0, 255) # 파란색: 잠재적 폐사
            cv2.rectangle(dead_chicken_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(dead_chicken_frame, f"Dead {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (0, 255, 0) # 초록색: 움직임 감지

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"Chicken {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Original Detection (FFT)", annotated_frame)
    cv2.imshow("Likely Dead Chickens (FFT)", dead_chicken_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()