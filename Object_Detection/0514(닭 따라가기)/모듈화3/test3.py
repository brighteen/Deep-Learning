import cv2
import os
import numpy as np
from ultralytics import YOLO

class IDManager:
    """객체 ID를 관리하는 간소화된 클래스"""
    
    def __init__(self):
        # ID 집합 관리 데이터
        self.id_sets = []  # 같은 개체로 판단된 ID 집합들
        self.id_to_set_index = {}  # ID가 어느 집합에 속하는지 매핑
        self.set_to_display_id = {} # 각 집합의 대표 ID
        self.id_to_box = {}  # 각 ID의 바운딩 박스 정보
        self.frame_count = 0  # 현재 프레임 번호
        
        # 설정
        self.distance_threshold = 15  # 같은 객체로 판단할 최대 거리
    
    def update(self, detected_objects):
        """
        탐지된 객체 정보 업데이트 및 ID 집합 관리
        
        Args:
            detected_objects: [(x1, y1, x2, y2, id), ...] 형태의 객체 목록
        """
        self.frame_count += 1
        
        # 탐지된 객체 정보 저장
        for obj in detected_objects:
            x1, y1, x2, y2, obj_id = obj
            box = (x1, y1, x2, y2)
            
            # ID가 유효한 경우에만 처리
            if obj_id != -1:
                # 바운딩 박스 정보 저장
                self.id_to_box[obj_id] = box
                
                # 해당 ID가 어떤 집합에 속하는지 확인
                if obj_id in self.id_to_set_index:
                    set_index = self.id_to_set_index[obj_id]
                else:
                    # 새 집합 생성
                    set_index = len(self.id_sets)
                    self.id_sets.append({obj_id})
                    self.id_to_set_index[obj_id] = set_index
                    self.set_to_display_id[set_index] = obj_id
        
        # ID 집합 병합 처리
        self._merge_id_sets(detected_objects)
    
    def _merge_id_sets(self, objects):
        """서로 가까운 객체들의 ID 집합을 병합"""
        # 각 객체 쌍에 대해 거리 계산
        for i, obj1 in enumerate(objects):
            x1_1, y1_1, x2_1, y2_1, id1 = obj1
            if id1 == -1:
                continue
                
            center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
            
            for j in range(i + 1, len(objects)):
                x1_2, y1_2, x2_2, y2_2, id2 = objects[j]
                if id2 == -1:
                    continue
                    
                # 이미 같은 집합인 경우 건너뜀
                if (id1 in self.id_to_set_index and id2 in self.id_to_set_index and 
                    self.id_to_set_index[id1] == self.id_to_set_index[id2]):
                    continue
                
                # 두 객체 간 거리 계산
                center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
                distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                
                # 거리가 임계값보다 가까우면 두 집합 병합
                if distance < self.distance_threshold:
                    set_index1 = self.id_to_set_index.get(id1)
                    set_index2 = self.id_to_set_index.get(id2)
                    
                    if set_index1 is not None and set_index2 is not None:
                        if set_index1 != set_index2:
                            # 작은 인덱스의 집합으로 병합
                            from_index = max(set_index1, set_index2)
                            to_index = min(set_index1, set_index2)
                            
                            # 집합 병합
                            self.id_sets[to_index].update(self.id_sets[from_index])
                            
                            # ID와 집합 인덱스 매핑 업데이트
                            for moved_id in self.id_sets[from_index]:
                                self.id_to_set_index[moved_id] = to_index
                            
                            # 대표 ID 업데이트 (작은 ID를 대표로 사용)
                            self.set_to_display_id[to_index] = min(
                                self.set_to_display_id.get(to_index, float('inf')),
                                self.set_to_display_id.get(from_index, float('inf'))
                            )
                            
                            # 병합된 집합 비우기
                            self.id_sets[from_index] = set()
                            self.set_to_display_id.pop(from_index, None)
                    
                    elif set_index1 is not None:
                        # id2를 id1의 집합에 추가
                        self.id_sets[set_index1].add(id2)
                        self.id_to_set_index[id2] = set_index1
                    
                    elif set_index2 is not None:
                        # id1을 id2의 집합에 추가
                        self.id_sets[set_index2].add(id1)
                        self.id_to_set_index[id1] = set_index2
    
    def get_display_id(self, obj_id):
        """객체의 대표 ID 반환"""
        if obj_id in self.id_to_set_index:
            set_index = self.id_to_set_index[obj_id]
            return self.set_to_display_id.get(set_index, obj_id)
        return obj_id
    
    def get_id_set(self, obj_id):
        """객체가 속한 ID 집합 반환"""
        if obj_id in self.id_to_set_index:
            set_index = self.id_to_set_index[obj_id]
            return self.id_sets[set_index]
        return {obj_id}

def main():
    # 파일 경로 설정
    video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"
    model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best.pt"
    
    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
        return
    
    # YOLO 모델 로드
    try:
        model = YOLO(model_path)
        print("YOLO 모델 로드 성공")
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        return
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return
    
    # 비디오 창 생성
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    
    # ID 관리자 초기화
    id_manager = IDManager()
    
    # 탐지 설정
    conf_threshold = 0.5
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("비디오 끝")
            break
        
        frame = frame[1000:1500, 200:1000]

        # 객체 탐지 (YOLOv8 추적 모드 사용)
        results = model.track(frame, conf=conf_threshold, persist=True)[0]
        
        # 결과 추출
        boxes = results.boxes.xyxy.cpu().tolist() if hasattr(results.boxes, 'xyxy') else []
        ids = results.boxes.id.int().cpu().tolist() if hasattr(results.boxes, 'id') else [-1] * len(boxes)
        
        # 탐지된 객체 정보
        detected_objects = [(box[0], box[1], box[2], box[3], id) for box, id in zip(boxes, ids)]
        
        # ID 관리자 업데이트
        id_manager.update(detected_objects)
        
        # 결과 시각화
        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = [int(p) for p in box]
            
            # 대표 ID 얻기
            if obj_id != -1:
                display_id = id_manager.get_display_id(obj_id)
                id_set = id_manager.get_id_set(obj_id)
                
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # ID 표시 (집합 정보 포함)
                if len(id_set) > 1:
                    label = f"{obj_id}→{display_id}"
                else:
                    label = f"{display_id}"
                
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 화면에 표시
        cv2.imshow("Object Detection", frame)
        
        # 키 입력 처리
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()