import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time

# 모델 로드
model = YOLO(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt")

# 비디오 파일 열기
cap = cv2.VideoCapture(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\detect2YOLO\datas\tile_r0_c1.mp4")

# 추적 중인 객체 정보 저장 (ID: [bbox, stillness_duration, initial_center])
tracked_objects = {}
object_id_counter = 0
stillness_duration_threshold = 30  # 움직임 없음 지속 시간 (초)
frame_rate = 30  # 초기 프레임 레이트 추정 (실제 값은 동적으로 계산)
not_moving_frames_threshold = stillness_duration_threshold * frame_rate
static_threshold = 20 # 움직임 없다고 판단하는 중심점 최대 이동 거리 (픽셀)
optical_flow_threshold = 0.5 # 광학 흐름 움직임 임계값 (조정 필요)

prev_frame_time = None
prev_gray = None # 이전 grayscale 프레임 저장 변수 추가

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
    center_points_frame = np.zeros_like(frame) # 각 닭들의 점만 표시할 프레임 생성
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 현재 프레임 grayscale 변환

    # 각 프레임에 대해 예측 수행
    results = model.predict(frame, imgsz=1280, conf=0.3, iou=0.5)
    current_detections = results[0].boxes.xyxy.cpu().numpy().astype(int)

    new_tracked_objects = {}
    current_detected_indices = list(range(len(current_detections)))
    matched_indices = set()

    for obj_id, (prev_bbox, stillness_duration, initial_center) in list(tracked_objects.items()):
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
            distance_from_initial = np.sqrt((current_center[0] - initial_center[0])**2 + (current_center[1] - initial_center[1])**2)

            is_still_by_center = distance_from_initial <= static_threshold
            is_still_by_optical_flow = True # 초기값 설정

            if prev_gray is not None:
                prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
                curr_x1, curr_y1, curr_x2, curr_y2 = current_bbox

                # 이전 bounding box 기준으로 영역 추출 (크기 불일치 방지)
                prev_roi_y1 = max(0, prev_y1)
                prev_roi_y2 = min(prev_gray.shape[0], prev_y2)
                prev_roi_x1 = max(0, prev_x1)
                prev_roi_x2 = min(prev_gray.shape[1], prev_x2)
                prev_roi = prev_gray[prev_roi_y1:prev_roi_y2, prev_roi_x1:prev_roi_x2]

                curr_roi_y1 = max(0, prev_y1) # 이전 bounding box의 y 좌표 사용
                curr_roi_y2 = min(gray.shape[0], prev_y2)
                curr_roi_x1 = max(0, prev_x1) # 이전 bounding box의 x 좌표 사용
                curr_roi_x2 = min(gray.shape[1], prev_x2)
                curr_roi = gray[curr_roi_y1:curr_roi_y2, curr_roi_x1:curr_roi_x2]

                # ROI 크기가 유효한지 확인
                if prev_roi.shape[0] > 0 and prev_roi.shape[1] > 0 and curr_roi.shape[0] > 0 and curr_roi.shape[1] > 0 and prev_roi.shape == curr_roi.shape:
                    # 광학 흐름 계산 (Farneback 알고리즘 사용)
                    flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    # 광학 흐름 벡터들의 크기 계산 및 평균 값으로 움직임 정도 판단
                    flow_magnitude = np.linalg.norm(flow, axis=2)
                    mean_flow = np.mean(flow_magnitude)

                    if mean_flow > optical_flow_threshold:
                        is_still_by_optical_flow = False
                else:
                    is_still_by_optical_flow = False # ROI가 유효하지 않으면 움직임이 있다고 간주

            if is_still_by_center and is_still_by_optical_flow:
                new_tracked_objects[obj_id] = [current_bbox, stillness_duration + time_elapsed, initial_center]
            else:
                # 움직임 감지: 초기 중심점 업데이트 및 시간 초기화
                new_tracked_objects[obj_id] = [current_bbox, 0.0, current_center]

            matched_indices.add(best_match_index)
            if best_match_index in current_detected_indices:
                current_detected_indices.remove(best_match_index)
        else:
            # 매칭 실패: 움직임 없음 시간 유지 (매칭 안됐으니 움직임 없다고 간주)
            new_tracked_objects[obj_id] = [prev_bbox, stillness_duration + time_elapsed, initial_center]

    # 새로 감지된 객체 추가
    for index in current_detected_indices:
        bbox = current_detections[index]
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        new_tracked_objects[object_id_counter] = [bbox, 0.0, center]
        object_id_counter += 1

    tracked_objects = new_tracked_objects

    # 결과 시각화 (원본 프레임)
    annotated_frame = original_frame.copy()
    for obj_id, (bbox, stillness_duration, initial_center) in tracked_objects.items():
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)  # 기본은 초록색 (움직이는 상태)
        label = f"Chicken {obj_id}"

        death_probability = min(100, int((stillness_duration // stillness_duration_threshold) * (100 / stillness_duration_threshold) * 10)) # 단순화된 확률 계산

        label += f" ({death_probability}%)"

        if death_probability >= 40:
            color = (0, 0, 255)  # 40% 이상 확률이면 파란색 (잠재적 폐사)
            cv2.rectangle(dead_chicken_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(dead_chicken_frame, f"Dead {obj_id} ({death_probability}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif stillness_duration > 0:
            color = (0, 255, 255) # 움직임이 잠시 멈춘 경우 노란색

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(annotated_frame, initial_center, 5, (255, 0, 0), -1) # 초기 중심점 시각화

        # 각 닭의 중심점을 세 번째 창에 그리기 (움직임 없는 경우만)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        point_color = color # 원본 영상에서의 색상과 동일하게 사용
        if stillness_duration > 0:
            cv2.circle(center_points_frame, (center_x, center_y), 5, point_color, -1)

    # 창 띄우기
    cv2.imshow("Original Detection", annotated_frame)
    cv2.imshow("Likely Dead Chickens", dead_chicken_frame)
    cv2.imshow("Chicken Center Points", center_points_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray # 현재 grayscale 프레임을 이전 프레임으로 저장

cap.release()
cv2.destroyAllWindows()