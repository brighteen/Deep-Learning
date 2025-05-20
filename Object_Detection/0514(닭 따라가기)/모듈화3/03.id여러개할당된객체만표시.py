import cv2
import os
import numpy as np
from ultralytics import YOLO
import time

class IDManager:
    """객체 ID를 관리하는 간소화된 클래스"""
    
    def __init__(self):
        # ID 집합 관리 데이터
        self.id_sets = []  # 같은 개체로 판단된 ID 집합들
        self.id_to_set_index = {}  # ID가 어느 집합에 속하는지 매핑
        self.set_to_display_id = {} # 각 집합의 대표 ID
        self.id_to_box = {}  # 각 ID의 바운딩 박스 정보
        self.frame_count = 0  # 현재 프레임 번호
        self.all_detected_ids = set()  # 이전에 탐지된 모든 ID
        
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
                # 모든 탐지된 ID 기록
                self.all_detected_ids.add(obj_id)
                
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
        
        # 한 번도 집합에 포함되지 않은 ID 처리
        self._ensure_all_ids_in_sets()
    
    def _ensure_all_ids_in_sets(self):
        """모든 탐지된 ID가 어떤 집합에 속하도록 보장"""
        for obj_id in self.all_detected_ids:
            if obj_id not in self.id_to_set_index:
                # 집합에 포함되지 않은 ID는 새 집합 생성
                set_index = len(self.id_sets)
                self.id_sets.append({obj_id})
                self.id_to_set_index[obj_id] = set_index
                self.set_to_display_id[set_index] = obj_id
    
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
    
    def save_id_sets_to_file(self, filename):
        """ID 집합을 파일로 저장 - 모든 ID 집합 포함"""
        # 모든 탐지된 ID가 집합에 포함되도록 보장
        self._ensure_all_ids_in_sets()
        
        # 최종 ID 집합 생성 (대표 ID를 키로 사용)
        final_id_sets = {}
        
        # 비어있지 않은 집합들만 처리
        for set_idx, id_set in enumerate(self.id_sets):
            if not id_set:  # 빈 집합은 무시
                continue
                
            # 대표 ID 가져오기
            display_id = self.set_to_display_id.get(set_idx, min(id_set))
            
            # 최종 결과에 추가
            if display_id in final_id_sets:
                # 이미 존재하는 대표 ID면 집합 합치기
                final_id_sets[display_id].update(id_set)
            else:
                # 새 대표 ID면 새로운 집합 생성
                final_id_sets[display_id] = set(id_set)
        
        # 파일에 결과 저장
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # 대표 ID 순서대로 정렬해서 저장
                for display_id in sorted(final_id_sets.keys()):
                    id_set = final_id_sets[display_id]
                    # 모든 ID 집합 저장 (크기 제한 없음)
                    id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
                    f.write(f"{display_id}: {id_set_str}\n")
            
            print(f"ID 집합이 성공적으로 저장되었습니다: {filename}")
            return True
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return False

def main():
    # 현재 스크립트 파일의 경로 가져오기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 결과 파일 경로 설정 (스크립트와 동일한 폴더에 저장)
    result_file = os.path.join(script_dir, "tracking_results.txt")
    
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
    
    # 비디오 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"비디오 FPS: {fps}")
    
    # 비디오 창 생성
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    
    # ID 관리자 초기화
    id_manager = IDManager()
    
    # 탐지 설정
    conf_threshold = 0.5
    
    # 시간 제한 설정 (1분)
    start_time = time.time()
    duration = 60  # 60초 (1분)
    
    # 프레임 카운터
    frame_count = 0
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("비디오 끝")
            break

        # frame = frame[1000:1500, 200:1000]
        frame = frame[100:1000, 500:1400]
            
        # 프레임 카운터 증가
        frame_count += 1
        
        # 현재 시간 확인
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # 1분 경과 확인
        if elapsed_time >= duration:
            print(f"설정한 시간({duration}초) 경과. 처리 종료.")
            break
        
        # 화면에 시간 정보 표시용
        remaining_time = duration - elapsed_time
        
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
                
                # ID 집합 크기에 따라 바운딩 박스 색상 결정
                if len(id_set) > 1:
                    # ID가 재할당된 객체는 빨간색으로 표시
                    box_color = (0, 0, 255)  # 빨간색(BGR)
                else:
                    # 일반 객체는 녹색
                    box_color = (0, 255, 0)  # 녹색(BGR)
                
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # ID 집합이 2개 이상인 경우에만 레이블 표시
                if len(id_set) > 1:
                    # 집합 형식으로 표시 {고유id, 새로 할당된 id...}
                    id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
                    
                    # 배경 박스 그리기 (텍스트 가독성 향상)
                    (text_width, text_height), _ = cv2.getTextSize(id_set_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, 
                                 (int(x1), int(y1) - text_height - 5), 
                                 (int(x1) + text_width + 5, int(y1)), 
                                 (0, 0, 150), -1)  # 어두운 빨간색 배경
                    
                    # 텍스트 표시
                    cv2.putText(frame, id_set_str, (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 시간 정보 표시
        cv2.putText(frame, f"남은 시간: {int(remaining_time)}초", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 화면에 표시
        cv2.imshow("Object Detection", frame)
        
        # 키 입력 처리 (q: 종료)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자가 종료했습니다.")
            break
    
    # 결과 저장
    print(f"ID 추적 결과를 저장합니다: {result_file}")
    id_manager.save_id_sets_to_file(result_file)
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()