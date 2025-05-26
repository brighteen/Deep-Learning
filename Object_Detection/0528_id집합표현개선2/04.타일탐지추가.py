import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
from collections import defaultdict

def tile_based_detection(image, model, tile_size=640, overlap=0.2, conf_threshold=0.7):
    """
    이미지를 타일로 나누어 각각 객체 탐지만 수행 (추적 제외)
    """
    h, w = image.shape[:2]
    all_predictions = []
    
    # 타일 간 겹침 계산
    stride = int(tile_size * (1 - overlap))
    
    # 이미지를 타일로 나누어 각각 탐지 수행
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 타일 경계 계산
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            
            # 마지막 타일이 너무 작으면 조정
            if x2 - x < tile_size * 0.5:
                x = max(0, x2 - tile_size)
            if y2 - y < tile_size * 0.5:
                y = max(0, y2 - tile_size)
            
            # 타일 추출
            tile = image[y:y2, x:x2]
            
            # 타일이 비어있거나 너무 작으면 건너뛰기
            if tile.size == 0 or tile.shape[0] < 32 or tile.shape[1] < 32:
                continue
            
            # 타일에서 객체 탐지만 수행 (추적 제외)
            results = model.predict(tile, conf=conf_threshold, verbose=False)
            
            # 결과가 있으면 좌표 조정하여 저장
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                        # 클래스 확인 (chick만 처리)
                        cls = int(boxes.cls[i].item())
                        cls_name = model.names[cls]
                        
                        if cls_name.lower() != 'chick':
                            continue
                        
                        # 박스 좌표 조정 (x, y 오프셋 적용)
                        adjusted_box = box.copy()
                        adjusted_box[0] += x  # x1 조정
                        adjusted_box[1] += y  # y1 조정
                        adjusted_box[2] += x  # x2 조정
                        adjusted_box[3] += y  # y2 조정
                        
                        # 신뢰도 정보 포함
                        conf = float(boxes.conf[i].item())
                        all_predictions.append({
                            'box': adjusted_box,
                            'conf': conf,
                            'cls': cls
                        })
    
    # 타일 간 중복 탐지 결과 처리 (거리 기반)
    if not all_predictions:
        return []
    
    # 중복 제거: 중심점이 너무 가까운 탐지 결과 병합
    final_predictions = []
    processed = set()
    
    for i, pred1 in enumerate(all_predictions):
        if i in processed:
            continue
            
        # 현재 예측과 중복되는 다른 예측들 찾기
        duplicates = [pred1]
        box1 = pred1['box']
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        
        for j, pred2 in enumerate(all_predictions[i+1:], i+1):
            if j in processed:
                continue
                
            box2 = pred2['box']
            center2_x = (box2[0] + box2[2]) / 2
            center2_y = (box2[1] + box2[3]) / 2
            
            # 중심점 간 거리 계산
            distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
            
            # 임계값 이내면 중복으로 판단
            if distance < 50:  # 50픽셀 이내
                duplicates.append(pred2)
                processed.add(j)
        
        # 중복된 탐지 결과 중 가장 신뢰도가 높은 것 선택
        best_pred = max(duplicates, key=lambda x: x['conf'])
        final_predictions.append(best_pred)
        processed.add(i)
    
    return final_predictions

def create_detection_for_tracking(predictions, image_shape):
    """
    타일 기반 탐지 결과를 YOLO 추적기에 맞는 형태로 변환
    """
    if not predictions:
        return None
    
    # 탐지 결과를 numpy 배열로 변환
    detections = []
    for pred in predictions:
        box = pred['box']
        conf = pred['conf']
        cls = pred['cls']
        # [x1, y1, x2, y2, conf, cls] 형태로 변환
        detections.append([box[0], box[1], box[2], box[3], conf, cls])
    
    return np.array(detections)

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

# 처리할 시간 범위 (초)
start_time = 324.0
end_time = 329.0

# 시간을 프레임 인덱스로 변환
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)
total_frames = end_frame - start_frame

# 1프레임 간격으로 처리
frame_interval = 1

# 타일 기반 탐지 설정
tile_size = 640
overlap = 0.2

# 결과 저장할 디렉토리 설정
output_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\0528_id집합표현개선2\results\9.Tile탐지적용"
os.makedirs(output_dir, exist_ok=True)

# 색상 설정 (B,G,R)
GREEN = (0, 255, 0)     # 계속 탐지되는 객체
RED = (0, 0, 255)       # 사라진 ID
BLUE = (255, 0, 0)      # 새로 탐지된 ID
BLACK = (0, 0, 0)       # 같은 객체로 판단된 ID (집합으로 관리)

# ID 집합 관리를 위한 딕셔너리
id_mappings = {}  
id_to_main = {}   
last_positions = {}  

# 결과 저장을 위한 비디오 설정
output_video_path = os.path.join(output_dir, "tile_tracking_result.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps/frame_interval, (width, height))

# 현재 프레임과 이전 프레임의 ID 집합
current_ids = set()
previous_ids = set()

# 추적기 상태 관리
tracker_initialized = False
frame_detections_history = []  # 최근 몇 프레임의 탐지 결과 저장

# ID 관리 함수들 (기존과 동일)
def find_all_related_ids(id, visited=None):
    if visited is None:
        visited = set()
    
    if id in visited:
        return visited
        
    visited.add(id)
    
    if id in id_mappings:
        for related_id in id_mappings[id]:
            find_all_related_ids(related_id, visited)
    
    if id in id_to_main:
        main_id = id_to_main[id]
        find_all_related_ids(main_id, visited)
        
    return visited

def merge_id_sets(id1, id2):
    all_related_ids = find_all_related_ids(id1) | find_all_related_ids(id2)
    main_id = min(all_related_ids)
    
    for id_val in all_related_ids:
        if id_val in id_mappings:
            del id_mappings[id_val]
    
    id_list = sorted(all_related_ids)
    id_list.remove(main_id)
    id_list.insert(0, main_id)
    
    id_mappings[main_id] = set(id_list)
    
    for id_val in all_related_ids:
        id_to_main[id_val] = main_id
        
    return main_id

def check_same_object(new_id, new_box, disappeared_ids):
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
        
        distance = np.sqrt((new_center_x - old_center_x)**2 + (new_center_y - old_center_y)**2)
        
        if distance < 150 and distance < min_distance:
            min_distance = distance
            closest_id = old_id
    
    return closest_id

def get_main_id(id):
    return id_to_main.get(id, id)

def format_id_set(id_set, main_id):
    id_list = list(id_set)
    if main_id in id_list:
        id_list.remove(main_id)
        id_list.insert(0, main_id)
    return set(id_list)

# 간단한 추적 로직 구현
next_id = 1
active_tracks = {}  # {id: {'box': box, 'age': age}}

def simple_tracking(current_detections, previous_detections):
    """간단한 거리 기반 추적"""
    global next_id, active_tracks
    
    tracked_results = []
    
    if not current_detections:
        return []
    
    if not previous_detections:
        # 첫 프레임인 경우 새 ID 할당
        for det in current_detections:
            tracked_results.append({
                'box': det['box'],
                'id': next_id,
                'conf': det['conf'],
                'cls': det['cls']
            })
            active_tracks[next_id] = {'box': det['box'], 'age': 0}
            next_id += 1
        return tracked_results
    
    # 현재 탐지와 이전 추적 결과 매칭
    used_prev_ids = set()
    
    for curr_det in current_detections:
        curr_box = curr_det['box']
        curr_center = [(curr_box[0] + curr_box[2])/2, (curr_box[1] + curr_box[3])/2]
        
        best_match_id = None
        min_distance = float('inf')
        
        # 이전 추적 결과와 거리 계산
        for prev_result in previous_detections:
            if prev_result['id'] in used_prev_ids:
                continue
                
            prev_box = prev_result['box']
            prev_center = [(prev_box[0] + prev_box[2])/2, (prev_box[1] + prev_box[3])/2]
            
            distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                             (curr_center[1] - prev_center[1])**2)
            
            if distance < 100 and distance < min_distance:  # 100픽셀 임계값
                min_distance = distance
                best_match_id = prev_result['id']
        
        if best_match_id is not None:
            # 기존 ID와 매칭됨
            tracked_results.append({
                'box': curr_det['box'],
                'id': best_match_id,
                'conf': curr_det['conf'],
                'cls': curr_det['cls']
            })
            used_prev_ids.add(best_match_id)
            active_tracks[best_match_id] = {'box': curr_det['box'], 'age': 0}
        else:
            # 새로운 객체
            tracked_results.append({
                'box': curr_det['box'],
                'id': next_id,
                'conf': curr_det['conf'],
                'cls': curr_det['cls']
            })
            active_tracks[next_id] = {'box': curr_det['box'], 'age': 0}
            next_id += 1
    
    return tracked_results

# 프레임 처리 시작
print(f"{start_frame}부터 {end_frame}까지 {frame_interval} 프레임 간격으로 처리 중...")
print(f"타일 기반 탐지 설정: 타일 크기={tile_size}, 겹침={overlap*100}%")

processed_count = 0
disappeared_ids = set()
previous_tracked_results = []

for frame_idx in range(start_frame, end_frame, frame_interval):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        print(f"프레임 {frame_idx} 읽기 실패")
        continue
    
    current_time = frame_idx / fps
    
    # 1단계: 타일 기반 탐지
    print(f"프레임 {frame_idx}: 타일 기반 탐지 수행 중...")
    tile_detections = tile_based_detection(frame, model, tile_size, overlap)
    
    # 2단계: 간단한 추적 적용
    tracked_results = simple_tracking(tile_detections, previous_tracked_results)
    
    # 결과 이미지 생성
    annotated_frame = frame.copy()
    
    # 이전 프레임에서 현재 프레임으로 넘어갈 때 ID 상태 업데이트
    previous_ids = current_ids.copy()
    current_ids = set()
    
    # 추적 결과 처리
    for result in tracked_results:
        id = result['id']
        box = result['box']
        conf = result['conf']
        x1, y1, x2, y2 = box.astype(int)
        
        current_ids.add(id)
        last_positions[id] = (x1, y1, x2, y2)
        
        # 새로 등장한 ID인지 확인
        if id not in previous_ids:
            matched_id = check_same_object(id, (x1, y1, x2, y2), disappeared_ids)
            
            if matched_id:
                main_id = merge_id_sets(id, matched_id)
                id_set = id_mappings.get(main_id, {id})
                id_set = format_id_set(id_set, main_id)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BLACK, 2)
                
                if len(id_set) > 1:
                    text = f"ID: {main_id}/{id_set}"
                else:
                    text = f"ID: {main_id}"
                    
                cv2.putText(annotated_frame, text, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
                
                if matched_id in disappeared_ids:
                    disappeared_ids.remove(matched_id)
            else:
                if id not in id_mappings and id not in id_to_main:
                    id_mappings[id] = {id}
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BLUE, 2)
                cv2.putText(annotated_frame, f"ID: {id} (New)", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2)
        else:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), GREEN, 2)
            
            main_id = get_main_id(id)
            id_set = id_mappings.get(main_id, {id})
            id_set = format_id_set(id_set, main_id)
            
            if len(id_set) > 1:
                text = f"ID: {main_id}/{id_set}"
            else:
                text = f"ID: {id}"
            
            cv2.putText(annotated_frame, text, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    
    # 사라진 ID 처리
    newly_disappeared = previous_ids - current_ids
    disappeared_ids.update(newly_disappeared)
    
    for disappeared_id in newly_disappeared:
        if disappeared_id in last_positions:
            x1, y1, x2, y2 = last_positions[disappeared_id]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), RED, 2)
            
            main_id = get_main_id(disappeared_id)
            id_set = id_mappings.get(main_id, {disappeared_id})
            id_set = format_id_set(id_set, main_id)
            
            if len(id_set) > 1:
                text = f"ID: {main_id}/{id_set} (Lost)"
            else:
                text = f"ID: {disappeared_id} (Lost)"
                
            cv2.putText(annotated_frame, text, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    
    # 프레임 정보 추가
    cv2.putText(annotated_frame, f"Frame: {frame_idx}, Time: {current_time:.1f}s", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(annotated_frame, f"Tile: {tile_size}x{tile_size}, Overlap: {overlap*100}%", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    mapping_text = "ID Mappings: "
    for main_id, id_set in list(id_mappings.items())[:3]:
        id_set = format_id_set(id_set, main_id)
        mapping_text += f"{main_id}:{id_set} "
    cv2.putText(annotated_frame, mapping_text, (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.putText(annotated_frame, f"Detected: {len(tracked_results)} objects", (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    out.write(annotated_frame)
    
    # 다음 프레임을 위해 현재 결과 저장
    previous_tracked_results = tracked_results
    
    processed_count += 1
    progress = processed_count / ((end_frame - start_frame) // frame_interval) * 100
    print(f"처리 중... {progress:.1f}% ({processed_count}/{(end_frame - start_frame) // frame_interval})")

# 자원 해제
cap.release()
out.release()

# 결과 저장
output_txt_path = os.path.join(output_dir, "tile_id_mappings.txt")
with open(output_txt_path, 'w', encoding='utf-8') as f:
    f.write("=== 타일 기반 객체 탐지 및 ID 매핑 결과 ===\n")
    f.write(f"타일 크기: {tile_size}x{tile_size}\n")
    f.write(f"겹침 비율: {overlap*100}%\n")
    f.write(f"처리된 프레임: {start_frame} ~ {end_frame}\n")
    f.write("=" * 50 + "\n\n")
    
    for main_id, id_set in sorted(id_mappings.items()):
        id_set = format_id_set(id_set, main_id)
        f.write(f"{main_id}: {id_set}\n")

print(f"처리 완료! 결과 비디오: {output_video_path}")
print(f"ID 매핑 결과: {output_txt_path}")
print(f"최종 ID 매핑: {id_mappings}")