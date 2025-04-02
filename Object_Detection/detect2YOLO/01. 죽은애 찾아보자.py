import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# 모델 로드
model = YOLO("best_chick.pt")

# 비디오 파일 열기
cap = cv2.VideoCapture(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\detect2YOLO\datas\tile_r0_c2.mp4")

prev_frame_detections = defaultdict(list)
current_frame_detections = defaultdict(list)
object_id_counter = 0
movement_threshold = 5  # 움직임이 없다고 판단하는 픽셀 거리

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 각 프레임에 대해 예측 수행
    results = model.predict(frame, imgsz=1280, conf=0.3, iou=0.5)

    # 현재 프레임의 객체 정보 초기화
    current_frame_detections.clear()
    detected_objects = results[0].boxes.xyxy.cpu().numpy().astype(int)
    current_centers = []
    current_bboxes = []

    for i, (x1, y1, x2, y2) in enumerate(detected_objects):
        current_centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        current_bboxes.append((x1, y1, x2, y2))
        current_frame_detections[i] = [(x1, y1, x2, y2), False] # False: 움직이는 상태

    if prev_frame_detections:
        matched_detections = {}
        for current_index, current_center in enumerate(current_centers):
            min_distance = float('inf')
            closest_prev_index = None

            for prev_index, prev_info in prev_frame_detections.items():
                prev_bbox, _ = prev_info
                prev_center_x = (prev_bbox[0] + prev_bbox[2]) // 2
                prev_center_y = (prev_bbox[1] + prev_bbox[3]) // 2
                distance = np.sqrt((current_center[0] - prev_center_x)**2 + (current_center[1] - prev_center_y)**2)

                if distance < min_distance and prev_index not in matched_detections.values():
                    min_distance = distance
                    closest_prev_index = prev_index

            if closest_prev_index is not None and min_distance < 50: # 매칭 거리 임계값 설정
                prev_bbox, _ = prev_frame_detections[closest_prev_index]
                prev_center_x = (prev_bbox[0] + prev_bbox[2]) // 2
                prev_center_y = (prev_bbox[1] + prev_bbox[3]) // 2
                movement = np.sqrt((current_center[0] - prev_center_x)**2 + (current_center[1] - prev_center_y)**2)
                if movement < movement_threshold:
                    current_frame_detections[current_index][1] = True # 움직임 없음 상태로 변경
                matched_detections[current_index] = closest_prev_index

    # 결과 시각화 (프레임에 박스 그리기)
    annotated_frame = frame.copy()
    for i, (bbox, not_moving) in current_frame_detections.items():
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if not not_moving else (0, 0, 255) # 움직이는 닭은 초록색, 움직임 없는 닭은 파란색
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        label = "Chicken" + (" (Not Moving)" if not_moving else "")
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 현재 프레임의 객체 정보를 이전 프레임 정보로 업데이트
    prev_frame_detections.clear()
    for i, detection_info in current_frame_detections.items():
        prev_frame_detections[i] = list(detection_info)


cap.release()
cv2.destroyAllWindows()