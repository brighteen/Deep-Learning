import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
from collections import defaultdict
'''
324~329초 구간을 5프레임(0.2초) 간격으로 처리.

객체 상태에 따라 다른 색상으로 표시:

초록색: 계속 탐지되는 객체
빨간색: 사라진 ID (탐지가 풀린 객체)
파란색: 새로 탐지된 객체
검은색: 같은 객체로 판단되어 집합으로 관리되는 ID
같은 객체로 판단하는 기준:

새로 나타난 ID와 사라진 ID 간의 위치 거리를 계산
100픽셀 이내의 거리에 있으면 같은 객체로 판단하고 ID 집합으로 관리
{대표ID: {ID1, ID2, ...}} 형태로 저장
사라진 ID는 마지막 위치에 빨간색 박스로 표시되고, 그 위치 정보는 새로운 ID와 매칭에 사용됨됨.
'''
# 모델 로드
model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best.pt"
model = YOLO(model_path)

# 비디오 로드
video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("오류: 영상을 열 수 없습니다.")
    exit()

# 영상 속성 확인
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"비디오 FPS: {fps}, 해상도: {width}x{height}")

# 관심 영역 설정
roi_y1, roi_y2 = 800, 1600  # 세로 범위
roi_x1, roi_x2 = 700, 1800  # 가로 범위

# 처리할 시간 범위 (초)
start_time = 324.0
end_time = 329.0

# 시간을 프레임 인덱스로 변환
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)
total_frames = end_frame - start_frame

# 1프레임 간격으로 처리
frame_interval = 1

# 결과 저장할 디렉토리 설정
output_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\0528_id집합표현개선2\results\매프레임추적5(iou=0.6)"
os.makedirs(output_dir, exist_ok=True)

# 색상 설정 (B,G,R)
GREEN = (0, 255, 0)     # 계속 탐지되는 객체
RED = (0, 0, 255)       # 사라진 ID
BLUE = (255, 0, 0)      # 새로 탐지된 ID
BLACK = (0, 0, 0)       # 같은 객체로 판단된 ID (집합으로 관리)

# ID 집합 관리를 위한 딕셔너리
id_mappings = {}  # {대표ID: {ID1, ID2, ...}} 형태로 관리
last_positions = {}  # 마지막으로 객체가 탐지된 위치 저장

# 결과 저장을 위한 비디오 설정
output_video_path = os.path.join(output_dir, "tracking_result.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps/frame_interval, (roi_x2-roi_x1, roi_y2-roi_y1))

# 현재 프레임과 이전 프레임의 ID 집합
current_ids = set()
previous_ids = set()

# 새로 나타난 ID가 사라진 ID와 같은 객체인지 판단하는 함수
def check_same_object(new_id, new_box, disappeared_ids):
    # 거리 기반으로 가장 가까운 사라진 ID 찾기
    min_distance = float('inf')
    closest_id = None
    
    new_center_x = (new_box[0] + new_box[2]) / 2
    new_center_y = (new_box[1] + new_box[3]) / 2
    
    for old_id in disappeared_ids:
        if old_id not in last_positions:
            continue
            
        old_box = last_positions[old_id]
        old_center_x = (old_box[0] + old_box[2]) / 2
        old_center_y = (old_box[1] + old_box[3]) / 2
        
        # 유클리드 거리 계산
        distance = np.sqrt((new_center_x - old_center_x)**2 + (new_center_y - old_center_y)**2)
        
        # 임계값 (100픽셀) 이내이면서 가장 가까운 ID 선택
        if distance < 100 and distance < min_distance:
            min_distance = distance
            closest_id = old_id
    
    return closest_id

# 프레임 처리 시작
print(f"{start_frame}부터 {end_frame}까지 {frame_interval} 프레임 간격으로 처리 중...")

processed_count = 0
disappeared_ids = set()  # 사라진 ID 추적

for frame_idx in range(start_frame, end_frame, frame_interval):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        print(f"프레임 {frame_idx} 읽기 실패")
        continue
    
    # 현재 시간 (초)
    current_time = frame_idx / fps
    
    # ROI 적용
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    
    # YOLO로 객체 탐지
    results = model.track(roi, persist=True, iou=0.3)  # persist=True로 추적 활성화, iou(default=0.45 -> 0.6)로 NMS 임계값 설정
    
    # 결과 이미지에 박스와 ID 그리기
    annotated_roi = roi.copy()
    
    # 이전 프레임에서 현재 프레임으로 넘어갈 때 ID 상태 업데이트
    previous_ids = current_ids.copy()
    current_ids = set()
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if hasattr(boxes, 'id') and boxes.id is not None:
            for i, (box, id) in enumerate(zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().numpy())):
                # 클래스 확인 (chick만 처리)
                cls = int(boxes.cls[i].item())
                cls_name = model.names[cls]
                
                # chick 클래스만 처리
                if cls_name.lower() != 'chick':
                    continue
                    
                id = int(id)
                x1, y1, x2, y2 = box.astype(int)
                
                # 현재 ID 목록에 추가
                current_ids.add(id)
                
                # 위치 정보 업데이트
                last_positions[id] = (x1, y1, x2, y2)
                
                # 새로 등장한 ID인지 확인
                if id not in previous_ids:
                    # 사라진 ID와 같은 객체인지 확인
                    matched_id = check_same_object(id, (x1, y1, x2, y2), disappeared_ids)
                    
                    if matched_id:
                        # 같은 객체로 판단됨 -> ID 집합으로 관리
                        if matched_id in id_mappings:
                            # 기존 집합에 추가
                            id_mappings[matched_id].add(id)
                        else:
                            # 새 집합 생성
                            id_mappings[matched_id] = {matched_id, id}
                        
                        # 검은색으로 표시
                        cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), BLACK, 2)
                        
                        # 대표 ID와 현재 ID 표시
                        text = f"ID: {matched_id}/{id}"
                        cv2.putText(annotated_roi, text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
                        
                        # 사라진 ID 목록에서 제거
                        if matched_id in disappeared_ids:
                            disappeared_ids.remove(matched_id)
                    else:
                        # 새로 등장한 객체 -> 파란색으로 표시
                        cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), BLUE, 2)
                        cv2.putText(annotated_roi, f"ID: {id} (New)", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2)
                else:
                    # 계속 탐지되는 객체 -> 초록색으로 표시
                    cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), GREEN, 2)
                    
                    # ID가 집합에 속하는지 확인
                    display_id = id
                    for main_id, id_set in id_mappings.items():
                        if id in id_set:
                            display_id = f"{main_id}/{id}"
                            break
                    
                    cv2.putText(annotated_roi, f"ID: {display_id}", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    
    # 사라진 ID 처리
    newly_disappeared = previous_ids - current_ids
    disappeared_ids.update(newly_disappeared)
    
    # 사라진 ID 표시 (빨간색)
    for disappeared_id in newly_disappeared:
        if disappeared_id in last_positions:
            x1, y1, x2, y2 = last_positions[disappeared_id]
            cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), RED, 2)
            cv2.putText(annotated_roi, f"ID: {disappeared_id} (Lost)", (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    
    # 프레임 정보 추가
    cv2.putText(annotated_roi, f"Frame: {frame_idx}, Time: {current_time:.1f}s", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 현재 ID 매핑 상태 표시
    mapping_text = "ID Mappings: "
    for main_id, id_set in list(id_mappings.items())[:3]:  # 처음 3개만 표시
        mapping_text += f"{main_id}:{id_set} "
    cv2.putText(annotated_roi, mapping_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 비디오에 프레임 추가
    out.write(annotated_roi)
    
    # 진행 상황 표시
    processed_count += 1
    progress = processed_count / ((end_frame - start_frame) // frame_interval) * 100
    print(f"처리 중... {progress:.1f}% ({processed_count}/{(end_frame - start_frame) // frame_interval})")
    
    # 이미지로도 저장 (10개마다 하나씩)
    if processed_count % 10 == 0:
        img_path = os.path.join(output_dir, f"frame_{frame_idx}_time_{current_time:.1f}s.jpg")
        cv2.imwrite(img_path, annotated_roi)

# 자원 해제
cap.release()
out.release()

print(f"처리 완료! 결과 비디오: {output_video_path}")
print(f"최종 ID 매핑: {id_mappings}")