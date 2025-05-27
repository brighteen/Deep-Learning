import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
from collections import defaultdict

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
output_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\0528_id집합표현개선2\results\12.집합정렬완성"
os.makedirs(output_dir, exist_ok=True)

# 색상 설정 (B,G,R)
GREEN = (0, 255, 0)     # 계속 탐지되는 객체
RED = (0, 0, 255)       # 사라진 ID
BLUE = (255, 0, 0)      # 새로 탐지된 ID
BLACK = (0, 0, 0)       # 같은 객체로 판단된 ID (집합으로 관리)

# ID 집합 관리를 위한 딕셔너리 - {대표ID: {id1, id2, ...}} 형태로 관리
id_mappings = {}  
id_to_main = {}   # {ID: 대표ID} 형태로 역매핑 관리
last_positions = {}  # 마지막으로 객체가 탐지된 위치 저장

# 결과 저장을 위한 비디오 설정
output_video_path = os.path.join(output_dir, "tracking_result.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps/frame_interval, (roi_x2-roi_x1, roi_y2-roi_y1))

# 현재 프레임과 이전 프레임의 ID 집합
current_ids = set()
previous_ids = set()

# 재귀적으로 모든 관련 ID 찾기 (전체 ID 체인 추적)
def find_all_related_ids(id, visited=None):
    if visited is None:
        visited = set()
    
    if id in visited:
        return visited
        
    visited.add(id)
    
    # id가 대표 ID인 경우
    if id in id_mappings:
        for related_id in id_mappings[id]:
            find_all_related_ids(related_id, visited)
    
    # id가 다른 집합에 속한 경우
    if id in id_to_main:
        main_id = id_to_main[id]
        find_all_related_ids(main_id, visited)
        
    return visited

# ID 집합 병합 함수 - 체인으로 연결된 모든 ID를 하나의 집합으로 병합
def merge_id_sets(id1, id2):
    # 두 ID와 관련된 모든 ID 찾기
    all_related_ids = find_all_related_ids(id1) | find_all_related_ids(id2)
    
    # 가장 작은 ID를 대표 ID로 선택
    main_id = min(all_related_ids)
    
    # 기존의 모든 관련 매핑 삭제
    for id_val in all_related_ids:
        if id_val in id_mappings:
            del id_mappings[id_val]
    
    # 새 집합 생성
    id_mappings[main_id] = all_related_ids.copy()
    
    # 역매핑 업데이트
    for id_val in all_related_ids:
        id_to_main[id_val] = main_id
        
    return main_id

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

# ID의 대표 ID 찾기 함수
def get_main_id(id):
    return id_to_main.get(id, id)  # 역매핑이 없으면 자기 자신 반환

# ID 집합을 정렬된 집합 형태의 문자열로 변환하는 함수
def format_id_set_as_sorted_set_string(id_set, main_id):
    """
    ID 집합을 정렬된 집합 형태의 문자열로 반환
    
    Args:
        id_set: ID 집합
        main_id: 대표 ID
    
    Returns:
        정렬된 집합 형태의 문자열 (예: "{90, 104}")
    """
    # 전체를 정렬
    sorted_list = sorted(id_set)
    
    # 대표 ID가 가장 작은 값이 아닌 경우 첫 번째로 이동
    if main_id in sorted_list and sorted_list[0] != main_id:
        sorted_list.remove(main_id)
        sorted_list.insert(0, main_id)
    
    # 집합 형태의 문자열로 변환
    if len(sorted_list) == 1:
        return f"{{{sorted_list[0]}}}"
    else:
        return "{" + ", ".join(map(str, sorted_list)) + "}"

# 콘솔 출력용 정렬된 집합 딕셔너리 생성 함수
def get_sorted_id_mappings_as_sets():
    """내부 저장용 id_mappings를 정렬된 집합 형태 문자열로 변환"""
    sorted_mappings = {}
    for main_id, id_set in sorted(id_mappings.items()):
        sorted_mappings[main_id] = format_id_set_as_sorted_set_string(id_set, main_id)
    return sorted_mappings

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
    results = model.track(roi, persist=True, conf=0.7, iou=0.6)  # persist=True로 추적 활성화
    
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
                        main_id = merge_id_sets(id, matched_id)
                        
                        # 대표 ID의 집합으로 표시
                        id_set = id_mappings.get(main_id, {id})
                        
                        # 검은색으로 표시
                        cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), BLACK, 2)
                        
                        # ID 집합 정보를 텍스트로 표시 (정렬된 집합 형태)
                        if len(id_set) > 1:  # 집합에 여러 ID가 있는 경우
                            sorted_set_str = format_id_set_as_sorted_set_string(id_set, main_id)
                            text = f"ID: {main_id}:{sorted_set_str}"
                        else:  # 집합에 하나의 ID만 있는 경우 (자기 자신)
                            text = f"ID: {main_id}"
                            
                        cv2.putText(annotated_roi, text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
                        
                        # 사라진 ID 목록에서 제거
                        if matched_id in disappeared_ids:
                            disappeared_ids.remove(matched_id)
                    else:
                        # 처음 탐지된 객체라면 개별 ID로 추가
                        if id not in id_mappings and id not in id_to_main:
                            id_mappings[id] = {id}
                        
                        # 새로 등장한 객체 -> 파란색으로 표시
                        cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), BLUE, 2)
                        cv2.putText(annotated_roi, f"ID: {id} (New)", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2)
                else:
                    # 계속 탐지되는 객체 -> 초록색으로 표시
                    cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), GREEN, 2)
                    
                    # ID가 집합에 속하는지 확인
                    main_id = get_main_id(id)
                    id_set = id_mappings.get(main_id, {id})
                    
                    # ID 표시 (정렬된 집합 형태)
                    if len(id_set) > 1:  # 집합에 여러 ID가 있는 경우
                        sorted_set_str = format_id_set_as_sorted_set_string(id_set, main_id)
                        text = f"ID: {main_id}:{sorted_set_str}"
                    else:  # 집합에 하나의 ID만 있는 경우
                        text = f"ID: {id}"
                    
                    cv2.putText(annotated_roi, text, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    
    # 사라진 ID 처리
    newly_disappeared = previous_ids - current_ids
    disappeared_ids.update(newly_disappeared)
    
    # 사라진 ID 표시 (빨간색)
    for disappeared_id in newly_disappeared:
        if disappeared_id in last_positions:
            x1, y1, x2, y2 = last_positions[disappeared_id]
            cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), RED, 2)
            
            # 사라진 ID의 대표 ID 확인
            main_id = get_main_id(disappeared_id)
            id_set = id_mappings.get(main_id, {disappeared_id})
            
            # ID 표시 (정렬된 집합 형태)
            if len(id_set) > 1:  # 집합에 여러 ID가 있는 경우
                sorted_set_str = format_id_set_as_sorted_set_string(id_set, main_id)
                text = f"ID: {main_id}:{sorted_set_str} (Lost)"
            else:  # 집합에 하나의 ID만 있는 경우
                text = f"ID: {disappeared_id} (Lost)"
                
            cv2.putText(annotated_roi, text, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    
    # 프레임 정보 추가
    cv2.putText(annotated_roi, f"Frame: {frame_idx}, Time: {current_time:.1f}s", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 현재 ID 매핑 상태 표시 (정렬된 집합 형태)
    mapping_text = "ID Mappings: "
    sorted_mappings = get_sorted_id_mappings_as_sets()
    for i, (main_id, sorted_set_str) in enumerate(list(sorted_mappings.items())[:3]):  # 처음 3개만 표시
        mapping_text += f"{main_id}:{sorted_set_str} "
    cv2.putText(annotated_roi, mapping_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 비디오에 프레임 추가
    out.write(annotated_roi)
    
    # 진행 상황 표시
    processed_count += 1
    progress = processed_count / ((end_frame - start_frame) // frame_interval) * 100
    print(f"처리 중... {progress:.1f}% ({processed_count}/{(end_frame - start_frame) // frame_interval})")

# 자원 해제
cap.release()
out.release()

# 결과 저장 - ID 매핑을 txt 파일로 저장 (정렬된 집합 형태)
output_txt_path = os.path.join(output_dir, "id_mappings.txt")
with open(output_txt_path, 'w') as f:
    # 각 매핑을 한 줄씩 저장 (정렬된 집합 형태)
    for main_id, id_set in sorted(id_mappings.items()):
        sorted_set_str = format_id_set_as_sorted_set_string(id_set, main_id)
        f.write(f"{main_id}: {sorted_set_str}\n")

# 정렬된 집합 형태로 콘솔 출력
sorted_mappings = get_sorted_id_mappings_as_sets()

print(f"처리 완료! 결과 비디오: {output_video_path}")
print(f"ID 매핑 결과: {output_txt_path}")
print(f"최종 ID 매핑 (정렬된 집합): {sorted_mappings}")