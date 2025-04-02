import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import defaultdict

# 모델 로드
model = YOLO("best_chick.pt")

# 비디오 파일 열기
cap = cv2.VideoCapture(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\detect2YOLO\datas\tile_r0_c1.mp4")

# 추적 중인 객체 정보 저장 (ID: [bbox, last_seen_frame, mhi])
tracked_objects = {}
object_id_counter = 0
mhi_duration = 30 # 모션 히스토리 지속 시간 (프레임 수)
min_motion_pixels = 5 # 움직임으로 간주할 최소 픽셀 수

prev_frame = None

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
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is None:
        prev_frame = current_frame_gray
        continue

    results = model.predict(frame, imgsz=1280, conf=0.3, iou=0.5)
    current_detections = results[0].boxes.xyxy.cpu().numpy().astype(int)

    new_tracked_objects = {}
    current_detected_indices = list(range(len(current_detections)))
    matched_indices = set()

    for obj_id, (prev_bbox, _, mhi) in list(tracked_objects.items()):
        best_match_index = -1
        max_iou = 0

        for i in current_detected_indices:
            current_bbox = current_detections[i]
            iou = calculate_iou(prev_bbox, current_bbox)
            if iou > max_iou:
                max_iou = iou
                best_match_index = i

        if best_match_index != -1:
            current_bbox = current_detections[best_match_index]
            new_tracked_objects[obj_id] = [current_bbox, frame.shape[0], mhi] # last_seen_frame 대신 frame height 임시 저장
            matched_indices.add(best_match_index)
            if best_match_index in current_detected_indices:
                current_detected_indices.remove(best_match_index)
        else:
            # 매칭 실패: MHI 정보 유지
            new_tracked_objects[obj_id] = [prev_bbox, frame.shape[0], mhi]

    for index in current_detected_indices:
        bbox = current_detections[index]
        mask = np.zeros_like(current_frame_gray)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
        mhi = np.zeros_like(current_frame_gray, dtype=np.float32)
        cv2.updateMotionHistory(mask, mhi, timeStamp=time.time(), duration=mhi_duration)
        new_tracked_objects[object_id_counter] = [bbox, frame.shape[0], mhi]
        object_id_counter += 1

    tracked_objects = new_tracked_objects

    annotated_frame = frame.copy()
    dead_chicken_frame = np.zeros_like(frame)

    for obj_id, (bbox, last_seen, mhi) in tracked_objects.items():
        x1, y1, x2, y2 = bbox
        motion_mask = np.uint8(mhi > 0)
        motion_pixels = np.sum(motion_mask[y1:y2, x1:x2])

        if motion_pixels < min_motion_pixels and last_seen < frame.shape[0] - (mhi_duration // 2): # 일정 시간 이상 움직임 없으면
            color = (0, 0, 255) # 파란색: 잠재적 폐사
            cv2.rectangle(dead_chicken_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(dead_chicken_frame, f"Dead {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (0, 255, 0) # 초록색: 움직임 감지

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"Chicken {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # MHI 시각화 (디버깅용)
        mhi_vis = np.uint8(cv2.normalize(mhi, None, 0, 255, cv2.NORM_MINMAX))
        cv2.imshow(f"MHI {obj_id}", mhi_vis)

    cv2.imshow("Original Detection (MHI)", annotated_frame)
    cv2.imshow("Likely Dead Chickens (MHI)", dead_chicken_frame)

    prev_frame = current_frame_gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()