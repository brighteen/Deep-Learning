import cv2
import os
import numpy as np
import time
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union

def calculate_center(box):
    """바운딩 박스의 중심 좌표를 계산합니다."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def calculate_distance(center1, center2):
    """두 중심점 간의 유클리디안 거리를 계산합니다."""
    return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

def calculate_size(box):
    """바운딩 박스의 넓이를 계산합니다."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height

def has_nearby_objects(box, all_boxes, threshold=50):
    """주어진 박스 주변에 다른 객체가 있는지 확인합니다."""
    center = calculate_center(box)
    
    for other_box in all_boxes:
        if box is other_box:  # 자기 자신은 건너뜀
            continue
            
        other_center = calculate_center(other_box)
        distance = calculate_distance(center, other_center)
        
        if distance < threshold:
            return True  # 주변에 다른 객체 발견
    
    return False  # 주변에 다른 객체 없음

def clean_up_stale_data(frame_count, max_frames, chicken_id_sets, id_to_set_index, id_to_last_frame, set_to_display_id, id_to_box=None, id_to_size=None):
    """오래된 추적 데이터를 정리합니다."""
    # 프레임 임계값 계산
    frame_threshold = frame_count - max_frames
    
    # 오래된 ID 식별
    stale_ids = [chicken_id for chicken_id, last_frame in id_to_last_frame.items() 
                 if last_frame <= frame_threshold]
    
    # 오래된 데이터가 없으면 빠르게 반환
    if not stale_ids:
        return
    
    # 영향을 받는 집합 추적
    affected_sets = set()
    
    # 오래된 데이터 삭제
    for chicken_id in stale_ids:
        set_index = id_to_set_index.pop(chicken_id, None)
        if set_index is not None and set_index < len(chicken_id_sets):
            chicken_id_sets[set_index].discard(chicken_id)
            # 집합이 변경되었음을 기록
            affected_sets.add(set_index)
        
        # 기타 정보 삭제
        id_to_last_frame.pop(chicken_id, None)
        
        # 새로 추가된 데이터 구조도 정리
        if id_to_box is not None:
            id_to_box.pop(chicken_id, None)
        if id_to_size is not None:
            id_to_size.pop(chicken_id, None)
    
    # 빈 집합 식별
    empty_sets = [idx for idx in affected_sets if not chicken_id_sets[idx]]
    
    # 빈 집합의 대표 ID 삭제
    for set_index in empty_sets:
        set_to_display_id.pop(set_index, None)
    
    # 빈 집합이 있는 경우 데이터 구조 재구성
    if empty_sets and chicken_id_sets:
        # 새로운 집합 구조 생성
        non_empty_indices = {}
        new_chicken_id_sets = []
        
        # 빈 집합 필터링 및 인덱스 매핑
        for i, id_set in enumerate(chicken_id_sets):
            if id_set:  # 비어있지 않은 집합만 유지
                non_empty_indices[i] = len(new_chicken_id_sets)
                new_chicken_id_sets.append(id_set)
        
        # 데이터 구조 업데이트
        chicken_id_sets.clear()
        chicken_id_sets.extend(new_chicken_id_sets)
        
        # ID와 집합 인덱스 매핑 업데이트
        new_id_to_set_index = {}
        for chicken_id, old_index in id_to_set_index.items():
            if old_index in non_empty_indices:
                new_id_to_set_index[chicken_id] = non_empty_indices[old_index]
        id_to_set_index.clear()
        id_to_set_index.update(new_id_to_set_index)
        
        # 대표 ID 매핑 업데이트
        new_set_to_display_id = {}
        for old_index, display_id in set_to_display_id.items():
            if old_index in non_empty_indices:
                new_set_to_display_id[non_empty_indices[old_index]] = display_id
        set_to_display_id.clear()
        set_to_display_id.update(new_set_to_display_id)

def main():
    """영상을 불러와 YOLO 모델로 객체 탐지를 수행하는 간단한 예제"""
    # 파일 경로 설정
    video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"
    model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best.pt"
    
    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
        return
      
    # YOLO 모델 로드
    try:
        print(f"모델 로드 시도 중: {model_path}")
        model = YOLO(model_path)
        print(f"YOLO 모델을 성공적으로 로드했습니다: {model_path}")
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()  # 상세한 오류 출력
        return
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 열기 실패 시
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보 출력
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 프레임 간 정확한 딜레이 계산 (최적화 1)
    delay = int(1000 / fps)
    print(f"영상 크기: {width}x{height}, FPS: {fps:.2f}")
      
    # 화면 창 생성
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Detection", int(width * 0.3), int(height * 0.3))
    
    # 탐지 설정
    conf_threshold = 0.5  # 탐지 확신도 임계값
    # 이전 프레임에서 탐지한 객체들의 정보 저장 (ID 추적용) (최적화 2)
    last_detections = []
    
    # ID 관리를 위한 자료구조 (ChickenDetector에서 가져온 핵심 로직)
    chicken_id_sets = []  # 같은 닭으로 판단된 ID의 집합들
    id_to_set_index = {}  # ID가 어느 집합에 속하는지 매핑
    set_to_display_id = {} # 각 집합의 대표 ID
    id_to_last_frame = {}  # ID가 마지막으로 나타난 프레임 번호
    id_to_box = {}        # ID별 현재 바운딩 박스
    id_to_size = {}       # ID별 크기 정보
    frame_count = 0  # 현재 프레임 번호
    
    # ID 추적 설정
    distance_threshold = 15  # 같은 객체로 판단할 최대 거리 (30에서 15로 감소)
    iou_threshold = 0.2  # IoU 임계값 (이 값보다 크면 다른 객체로 구분)
    size_difference_threshold = 1.5  # 크기 차이 임계값 (이 값보다 크면 다른 객체로 구분)
    max_frames = 100  # 오래된 데이터 정리 기준 프레임 수
    
    # FPS 계산을 위한 변수
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0
    
    while True:        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("비디오의 끝에 도달했습니다.")
            break
          # 프레임 카운터 증가
        frame_count += 1
        current_time = time.time()
        
        # FPS 계산
        fps_frame_count += 1
        if current_time - fps_start_time >= 1.0:  # 1초마다 FPS 계산
            fps_display = fps_frame_count / (current_time - fps_start_time)
            fps_frame_count = 0
            fps_start_time = current_time
        
        # 관심 영역 설정 (필요한 부분만 잘라서 사용)
        frame = frame[1000:1500, 200:1000]
        
        # 객체 탐지 수행 - track 메서드 사용 (최적화 3)
        results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)[0]
        
        # 탐지 결과 데이터 추출 (최적화 4)
        boxes = results.boxes.xyxy.cpu().tolist() if hasattr(results.boxes, 'xyxy') else []
        ids = results.boxes.id.int().cpu().tolist() if hasattr(results.boxes, 'id') and results.boxes.id is not None else [-1] * len(boxes)
          # 100프레임마다 오래된 데이터 정리
        if frame_count % max_frames == 0:
            clean_up_stale_data(frame_count, max_frames, chicken_id_sets, id_to_set_index, id_to_last_frame, set_to_display_id, id_to_box, id_to_size)
          # 현재 프레임에서 탐지된 객체들의 정보 저장
        new_detections = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            obj_id = ids[i]
            
            # ID가 없을 경우 IOU로 ID 복원 (최적화 5)
            if obj_id == -1:
                for last_box, last_id in last_detections:
                    iou = calculate_iou(box, last_box)
                    if iou > 0.5:
                        obj_id = last_id
                        break
              # 추적 결과 저장
            new_detections.append(((x1, y1, x2, y2), obj_id))            # ID 집합 관리 로직 - 같은 닭으로 판단된 ID 집합 처리
            if obj_id != -1:
                # ID가 이미 어떤 집합에 속해 있는지 확인
                if obj_id in id_to_set_index:
                    set_index = id_to_set_index[obj_id]
                    
                    # 이전 바운딩 박스와 현재 바운딩 박스의 특성 비교
                    if obj_id in id_to_box:
                        prev_box = id_to_box[obj_id]
                        prev_size = id_to_size[obj_id]
                        curr_size = calculate_size(box)
                        
                        # IoU와 크기 비율 계산
                        iou_value = calculate_iou(prev_box, box)
                        size_ratio = max(prev_size, curr_size) / min(prev_size, curr_size) if min(prev_size, curr_size) > 0 else float('inf')
                        
                        # IoU와 크기 비율을 확인하지만, 특성이 달라도 ID 분리는 하지 않음
                        # (기존 집합을 유지하여 ID 변화 히스토리를 보존)
                else:
                    # 새로운 집합 생성
                    set_index = len(chicken_id_sets)
                    chicken_id_sets.append({obj_id})
                    id_to_set_index[obj_id] = set_index
                    set_to_display_id[set_index] = obj_id  # 대표 ID로 설정
                
                # 마지막 등장 프레임 정보 업데이트
                id_to_last_frame[obj_id] = frame_count
                
                # 바운딩 박스 및 크기 정보 업데이트
                id_to_box[obj_id] = box
                id_to_size[obj_id] = calculate_size(box)
                # 마지막으로 나타난 프레임 번호 갱신
                id_to_last_frame[obj_id] = frame_count
          # 오래된 데이터 정리
        clean_up_stale_data(frame_count, max_frames, chicken_id_sets, id_to_set_index, id_to_last_frame, set_to_display_id, id_to_box, id_to_size)
        
        # 현재 프레임의 객체들 간의 ID 집합 병합 수행
        # 각 객체 쌍에 대해 거리를 계산하고 가까운 경우 동일한 닭으로 간주
        for i in range(len(new_detections)):
            box1, id1 = new_detections[i]
            if id1 == -1:  # 유효한 ID가 없는 경우 건너뜀
                continue
                
            center1 = calculate_center(box1)
            
            for j in range(i + 1, len(new_detections)):
                box2, id2 = new_detections[j]
                if id2 == -1:  # 유효한 ID가 없는 경우 건너뜀
                    continue
                    
                # 이미 같은 집합인 경우 건너뜀
                if id1 in id_to_set_index and id2 in id_to_set_index and id_to_set_index[id1] == id_to_set_index[id2]:
                    continue
                    
                # 거리 계산
                center2 = calculate_center(box2)
                distance = calculate_distance(center1, center2)                # 객체 크기 계산
                size1 = calculate_size(box1)
                size2 = calculate_size(box2)
                
                # IoU 계산
                iou_value = calculate_iou(box1, box2)
                
                # 크기 비율 계산 (큰 값 / 작은 값)
                size_ratio = max(size1, size2) / min(size1, size2) if min(size1, size2) > 0 else float('inf')
                
                # 주변에 다른 객체가 있는지 확인
                is_crowded1 = has_nearby_objects(box1, [box for box, _ in new_detections if box != box1], 50)
                is_crowded2 = has_nearby_objects(box2, [box for box, _ in new_detections if box != box2], 50)
                
                # 혼잡한 지역에서는 더 엄격한 기준 적용
                effective_distance_threshold = distance_threshold
                if is_crowded1 or is_crowded2:
                    effective_distance_threshold *= 0.5  # 혼잡한 지역에서는 거리 임계값을 절반으로 줄임
                
                # 거리가 임계값보다 작고, IoU가 임계값보다 작고, 크기 차이가 임계값보다 작은 경우에만 같은 닭으로 간주
                if (distance < effective_distance_threshold and 
                    iou_value < iou_threshold and 
                    size_ratio < size_difference_threshold):
                    # 두 ID가 속한 집합 확인
                    set_index1 = id_to_set_index.get(id1)
                    set_index2 = id_to_set_index.get(id2)
                    
                    if set_index1 is not None and set_index2 is not None:
                        # 두 ID가 모두 어떤 집합에 속해 있는 경우, 두 집합을 병합
                        if set_index1 != set_index2:
                            # 작은 인덱스의 집합으로 병합 (일관성 유지)
                            from_index = max(set_index1, set_index2)
                            to_index = min(set_index1, set_index2)
                            
                            # 모든 ID를 통합 집합으로 이동
                            from_set = chicken_id_sets[from_index]
                            chicken_id_sets[to_index].update(from_set)
                            
                            # 이동된 ID들의 집합 인덱스 업데이트
                            for moved_id in from_set:
                                id_to_set_index[moved_id] = to_index
                                
                            # 대표 ID 업데이트 (작은 ID를 대표로 사용)
                            set_to_display_id[to_index] = min(
                                set_to_display_id.get(to_index, float('inf')),
                                set_to_display_id.get(from_index, float('inf'))
                            )
                            
                            # 병합된 집합 비우기 (나중에 정리)
                            chicken_id_sets[from_index] = set()
                            # 옵션: 병합된 집합의 대표 ID 제거
                            set_to_display_id.pop(from_index, None)
                    elif set_index1 is not None:
                        # id2만 새로운 경우, id1의 집합에 추가
                        chicken_id_sets[set_index1].add(id2)
                        id_to_set_index[id2] = set_index1
                        # 대표 ID 업데이트 (작은 ID 사용)
                        set_to_display_id[set_index1] = min(set_to_display_id[set_index1], id2)
                    elif set_index2 is not None:
                        # id1만 새로운 경우, id2의 집합에 추가
                        chicken_id_sets[set_index2].add(id1)
                        id_to_set_index[id1] = set_index2
                        # 대표 ID 업데이트 (작은 ID 사용)
                        set_to_display_id[set_index2] = min(set_to_display_id[set_index2], id1)        # 직접 시각화 방식 사용 (results.plot() 대신 직접 그리기로 최적화 6)
        detected_frame = frame.copy()
        for (x1, y1, x2, y2), obj_id in new_detections:
            # 바운딩 박스 그리기 (모든 객체는 녹색으로 통일)
            cv2.rectangle(detected_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)              # ID 집합 정보를 표시 (간소화된 버전)
            if obj_id != -1 and obj_id in id_to_set_index:
                set_index = id_to_set_index[obj_id]
                display_id = set_to_display_id.get(set_index, obj_id)  # 대표(고유) ID
                  # 대표 ID와 ID 집합 정보를 표시
                if set_index < len(chicken_id_sets):
                    id_set = chicken_id_sets[set_index]
                    
                    # ID 집합 표시 - ID 변화가 있는 경우 집합으로 표시
                    if len(id_set) > 1:  # 집합에 여러 ID가 있는 경우 (탐지가 풀렸다 다시 잡힌 경우)
                        # ID 집합에 현재 ID가 포함되어 있는지 확인
                        current_in_set = obj_id in id_set
                        
                        # 새롭게 부여받은 ID인지 확인 (현재 표시 중인 객체의 ID와 대표 ID가 다른 경우)
                        is_reassigned = (obj_id != display_id)
                        
                        # ID 집합 문자열 생성
                        if len(id_set) <= 5:  # ID가 5개 이하면 모두 표시
                            id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
                        else:  # ID가 많으면 처음과 마지막만 표시
                            sorted_ids = sorted(id_set)
                            id_set_str = f"{{{sorted_ids[0]}...{sorted_ids[-1]}}}"
                        
                        # 현재 객체가 새 ID로 다시 탐지된 경우 강조 표시
                        if is_reassigned:
                            # 새 ID -> 대표 ID로 변환되었음을 표시
                            label = f"{obj_id}->{display_id} {id_set_str}"
                        else:
                            # 대표 ID와 ID 집합 표시
                            label = f"{display_id} {id_set_str}"
                    else:
                        # 단일 ID인 경우 간단히 표시
                        label = f"{display_id}"
                else:
                    label = f"{obj_id}"
            else:
                label = f"{obj_id}" if obj_id != -1 else "?"
            
            # 배경 박스로 텍스트 가독성 개선
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(detected_frame, 
                         (int(x1), int(y1) - text_height - 5), 
                         (int(x1) + text_width + 5, int(y1)), 
                         (0, 0, 0), 
                         -1)  # 검은색 배경
            
            # 텍스트 표시 (모든 텍스트는 흰색으로 통일)
            cv2.putText(detected_frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 탐지된 객체 수 계산
        chicken_count = len(new_detections)        # 화면에 FPS만 표시 (간소화)
        cv2.putText(detected_frame, f"FPS: {fps_display:.1f}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 키 도움말 표시
        cv2.putText(detected_frame, "Press 'q' to quit", (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 화면에 표시
        cv2.imshow("Object Detection", detected_frame)
        
        # 현재 프레임 정보를 다음 프레임에서 사용하기 위해 저장 (최적화 7)
        last_detections = [((x1, y1, x2, y2), obj_id) for (x1, y1, x2, y2), obj_id in new_detections]
        
        # 키 입력 처리 (q: 종료, 스페이스바: 일시정지)
        # 정확한 딜레이 적용 (최적화 8)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            print("사용자가 종료했습니다.")
            break
        elif key == ord(' '):
            print("일시정지됨. 계속하려면 아무 키나 누르세요.")
            cv2.waitKey(0)
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()